"""
Central configuration for the SmarTraffic DCRNN project.

All hyperparameters, paths, and runtime settings live here.
Import this module everywhere instead of hardcoding values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Dataset registry ──────────────────────────────────────────────────────────
DATASET_REGISTRY: dict[str, dict] = {
    "metr-la": {
        "url": "https://drive.usercontent.google.com/download?id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC",
        "adj_url": "https://drive.usercontent.google.com/download?id=1ous_RFaTvVA8j1-uLxL9ORQA7dFfLzSq",
        "raw_h5": DATA_DIR / "metr-la" / "metr-la.h5",
        "adj_pkl": DATA_DIR / "metr-la" / "adj_mx.pkl",
        "num_nodes": 207,
        "num_edges": 1722,
    },
    "pems-bay": {
        "url": "https://drive.usercontent.google.com/download?id=1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq",
        "adj_url": "https://drive.usercontent.google.com/download?id=1ous_RFaTvVA8j1-uLxL9ORQA7dFfLzSq",
        "raw_h5": DATA_DIR / "pems-bay" / "pems-bay.h5",
        "adj_pkl": DATA_DIR / "pems-bay" / "adj_mx_bay.pkl",
        "num_nodes": 325,
        "num_edges": 2694,
    },
}


@dataclass
class DataConfig:
    dataset: Literal["metr-la", "pems-bay"] = "metr-la"
    seq_len: int = 12          # input sequence length (1 h at 5-min intervals)
    horizon: int = 12          # prediction horizon (1 h = 60 min)
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    normalize: bool = True     # Z-score normalization per feature


@dataclass
class GraphConfig:
    sigma2: float = 0.1        # Gaussian kernel bandwidth for adjacency
    epsilon: float = 0.5       # sparsity threshold
    num_diffusion_steps: int = 2  # K in diffusion convolution


@dataclass
class ModelConfig:
    # DCRNN dimensions
    input_dim: int = 2         # flow + speed (+ optional occupancy)
    output_dim: int = 1        # predict flow only
    rnn_units: int = 64        # hidden size per GRU cell
    num_rnn_layers: int = 2    # stacked encoder/decoder layers
    max_diffusion_step: int = 2
    cl_decay_steps: int = 2000 # curriculum learning annealing steps
    use_curriculum_learning: bool = True
    filter_type: Literal["dual_random_walk", "laplacian", "random_walk"] = "dual_random_walk"


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    lr_milestones: list[int] = field(default_factory=lambda: [20, 30, 40, 50])
    lr_decay_ratio: float = 0.1
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    patience: int = 15         # early stopping
    log_every: int = 10        # log every N batches
    val_every: int = 1         # validate every N epochs
    save_dir: Path = MODEL_DIR
    device: str = "auto"       # "auto" | "cpu" | "cuda" | "mps"
    seed: int = 42
    amp: bool = True           # automatic mixed precision


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    ws_interval_s: float = 5.0  # push predictions every N seconds
    map_center: tuple[float, float] = (34.0195, -118.4912)  # LA downtown
    map_zoom: int = 11


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ── Convenience singleton ──────────────────────────────────────────────────────
cfg = Config()
