"""Configuration. Command line arguments override values from the yaml file."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # data
    data_dir: str = "data/features"
    top_poses: int = 500                  # keep the top N poses by docking score
    multi_apo: bool = True                # sample an apo structure per step
    score_only_test: bool = True          # score the flagged test units only

    # model
    dim: int = 48
    heads: int = 4
    blocks: int = 2
    dropout: float = 0.1
    res_dropout: float = 0.2

    # loss
    use_delta_rank: bool = True           # auxiliary ranking loss
    lambda_delta: float = 0.125
    rank_margin: float = 0.1

    # map
    map_tau: float = 1.0                  # softmax temperature over poses

    # optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_units: int = 8
    epochs: int = 400
    patience: int = 30
    grad_clip: float = 1.0
    seed: int = 0
    gpu: int = 0

    # output
    out_dir: str = "runs"
    tag: str = "epipose"
    ckpt: str = ""                        # checkpoint for eval and predict

    @classmethod
    def load(cls, path: str | None = None, **overrides) -> "Config":
        values: dict = {}
        if path:
            with open(path) as f:
                values.update(yaml.safe_load(f) or {})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))
