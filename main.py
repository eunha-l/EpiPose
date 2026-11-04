#!/usr/bin/env python
"""EpiPose entry point.

  python main.py train   --config configs/train.yaml
  python main.py train   --config configs/train_no_aux.yaml
  python main.py eval    --config configs/predict.yaml --ckpt checkpoints/epipose_seed0.pt
  python main.py predict --config configs/predict.yaml --ckpt ... --split test

Any value in the yaml file can be overridden by the command line argument of
the same name.
"""
from __future__ import annotations

import argparse
import sys

from epipose.config import Config
from epipose import runner


def _add_overrides(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to the yaml config")
    p.add_argument("--data_dir")
    p.add_argument("--top_poses", type=int)
    p.add_argument("--gpu", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--tag")
    p.add_argument("--out_dir")
    p.add_argument("--ckpt")
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--lambda_delta", type=float)
    p.add_argument("--rank_margin", type=float)
    p.add_argument("--map_tau", type=float)
    p.add_argument("--use_delta_rank", dest="use_delta_rank",
                   action="store_const", const=True, default=None,
                   help="train with the auxiliary ranking loss")
    p.add_argument("--no_delta_rank", dest="use_delta_rank",
                   action="store_const", const=False,
                   help="train with the pose loss only")
    p.add_argument("--multi_apo", dest="multi_apo", action="store_const",
                   const=True, default=None)
    p.add_argument("--no_multi_apo", dest="multi_apo", action="store_const",
                   const=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EpiPose")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("train", "train a model"),
                           ("eval", "evaluate a checkpoint"),
                           ("predict", "write predicted maps")):
        sp = sub.add_parser(name, help=helptext)
        _add_overrides(sp)
        if name == "predict":
            sp.add_argument("--split", default="test",
                            choices=("train", "val", "test"))
            sp.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    over = {k: v for k, v in vars(args).items()
            if k not in ("cmd", "config", "split", "out")}
    cfg = Config.load(args.config, **over)

    if args.cmd == "train":
        runner.train(cfg)
    elif args.cmd == "eval":
        if not cfg.ckpt:
            ap.error("eval needs --ckpt")
        runner.evaluate_ckpt(cfg)
    else:
        if not cfg.ckpt:
            ap.error("predict needs --ckpt")
        runner.predict(cfg, args.split, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
