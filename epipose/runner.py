"""Training, evaluation and prediction."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data import FeatureStore
from .losses import delta_rank, pose_bce
from .metrics import evaluate, pose_auroc
from .model import EpiPose, to_map


def build(cfg: Config):
    dev = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    store = FeatureStore(cfg.data_dir, dev, top_poses=cfg.top_poses,
                         use_multi_apo=cfg.multi_apo,
                         use_delta=cfg.use_delta_rank,
                         score_only_test=cfg.score_only_test)
    net = EpiPose(store.n_pose_feat, store.n_res_feat, dim=cfg.dim, heads=cfg.heads,
                  blocks=cfg.blocks, drop=cfg.dropout,
                  res_drop=cfg.res_dropout).to(dev)
    return store, net, dev


def _forward_unit(net, store, gid: str, train: bool):
    x, r, mask, views = store.batch([gid], train)
    logit = net(x, r, mask)[0]
    return logit, views[0]


@torch.no_grad()
def predict_split(net, store, cfg: Config, split: str):
    """Returns (gids, maps, labels, pose scores, pose qualities) for one split."""
    net.eval()
    maps, labels, logits, qs, gids = [], [], [], [], []
    for gid in store.by_split[split]:
        logit, view = _forward_unit(net, store, gid, train=False)
        maps.append(to_map(logit, view["contact"], cfg.map_tau).cpu().numpy())
        labels.append(store.units[gid].y.cpu().numpy())
        logits.append(logit.cpu().numpy())
        qs.append(view["q"].cpu().numpy())
        gids.append(gid)
    return gids, maps, labels, logits, qs


def _epoch(net, store, cfg, opt, split: str, train: bool):
    gids = list(store.by_split[split])
    if train:
        gids = [gids[i] for i in np.random.permutation(len(gids))]
    net.train(train)
    total = n_batch = 0.0
    for i in range(0, len(gids), cfg.batch_units):
        chunk = gids[i:i + cfg.batch_units]
        with torch.set_grad_enabled(train):
            loss = 0.0
            for gid in chunk:
                logit, view = _forward_unit(net, store, gid, train)
                loss = loss + pose_bce(logit, view["q"])
                delta = store.units[gid].delta
                if cfg.use_delta_rank and delta is not None:
                    loss = loss + cfg.lambda_delta * delta_rank(
                        logit, view["contact"], delta, cfg.rank_margin)
            loss = loss / len(chunk)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt.step()
        total += float(loss.detach())
        n_batch += 1
    return total / max(n_batch, 1)


def train(cfg: Config):
    store, net, dev = build(cfg)
    out = Path(cfg.out_dir) / cfg.tag
    out.mkdir(parents=True, exist_ok=True)
    cfg.dump(out / "config.yaml")
    print(store.summary(), flush=True)
    print(f"  {sum(p.numel() for p in net.parameters()):,} parameters, "
          f"auxiliary loss "
          f"{'on (lambda=%g)' % cfg.lambda_delta if cfg.use_delta_rank else 'off'}",
          flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    best = (-9.9, None, -1)
    hist = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        tl = _epoch(net, store, cfg, opt, "train", True)
        vl = _epoch(net, store, cfg, opt, "val", False)
        _, maps, labels, logits, qs = predict_split(net, store, cfg, "val")
        val = evaluate(list(zip(maps, labels)))
        pa, _ = pose_auroc(logits, qs)
        if val["auroc"] > best[0]:
            best = (val["auroc"],
                    {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
            torch.save(best[1], out / "best.pt")
        hist.append(dict(epoch=ep, train_loss=tl, val_loss=vl, pose_auroc=pa,
                         **{k: v for k, v in val.items()}))
        if ep % 5 == 0 or ep == best[2]:
            print(f"  ep {ep:3d} loss {tl:.4f} val {vl:.4f} | val AUROC "
                  f"{val['auroc']:.4f} MCC {val['mcc']:.4f} "
                  f"pose AUC {pa:.4f}  {time.time() - t0:.0f}s"
                  f"{'  *' if ep == best[2] else ''}", flush=True)
        if ep - best[2] >= cfg.patience:
            print(f"  early stop (best epoch {best[2]})", flush=True)
            break

    net.load_state_dict(best[1])
    val_res, test_res = _final_report(net, store, cfg)
    json.dump(dict(config=cfg.tag, best_epoch=best[2], val=val_res,
                   test=test_res, history=hist),
              open(out / "result.json", "w"), indent=1)
    return net, store, test_res


def _final_report(net, store, cfg):
    _, vmaps, vlab, _, _ = predict_split(net, store, cfg, "val")
    val_res = evaluate(list(zip(vmaps, vlab)))
    _, tmaps, tlab, tlogit, tq = predict_split(net, store, cfg, "test")
    test_res = evaluate(list(zip(tmaps, tlab)), val_res["threshold"])
    pa, skipped = pose_auroc(tlogit, tq)
    test_res["pose_auroc"] = pa
    test_res["pose_auroc_skipped"] = skipped
    print("", flush=True)
    print(f"{'map':16s}{'AUROC':>9s}{'AUPRC':>9s}{'MCC':>9s}{'F1':>9s}{'F1@30':>9s}",
          flush=True)
    print(f"{'EpiPose':16s}{test_res['auroc']:9.4f}{test_res['auprc']:9.4f}"
          f"{test_res['mcc']:9.4f}{test_res['f1']:9.4f}{test_res['f1_at_k']:9.4f}",
          flush=True)
    print(f"  pose AUROC {pa:.4f}  ({skipped} units skipped)", flush=True)
    return val_res, test_res


def load_checkpoint(cfg: Config):
    store, net, dev = build(cfg)
    sd = torch.load(cfg.ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    net.load_state_dict(sd)
    return store, net


def evaluate_ckpt(cfg: Config):
    store, net = load_checkpoint(cfg)
    print(store.summary(), flush=True)
    return _final_report(net, store, cfg)


def predict(cfg: Config, split: str, out_path: str | None):
    """Writes the predicted maps and pose scores to a npz file."""
    store, net = load_checkpoint(cfg)
    gids, maps, labels, logits, qs = predict_split(net, store, cfg, split)
    out_path = out_path or f"{cfg.tag}_{split}_pred.npz"
    np.savez_compressed(out_path,
                        gids=np.array(gids),
                        **{f"map_{g}": m for g, m in zip(gids, maps)},
                        **{f"pose_logit_{g}": l for g, l in zip(gids, logits)})
    print(f"wrote {out_path} for {len(gids)} units", flush=True)
    return gids, maps
