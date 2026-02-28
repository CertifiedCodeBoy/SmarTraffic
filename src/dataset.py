"""
Dataset loading and preprocessing for METR-LA / PeMS-BAY.

Key design decisions
--------------------
* Data is read once from HDF5 and kept as memory-mapped numpy arrays to keep
  RAM usage predictable.
* The StandardScaler is fit **only** on the training split to prevent leakage.
* `TrafficDataset` returns (x, y) tensors where:
    x : (seq_len, N, F)   – input window
    y : (horizon, N, F)   – prediction target
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import cfg, DATASET_REGISTRY, DataConfig
from src.utils import StandardScaler, logger


# ── Raw data loading ──────────────────────────────────────────────────────────

def load_traffic_h5(path: Path) -> pd.DataFrame:
    """Load METR-LA / PeMS-BAY HDF5 file into a DataFrame (T × N)."""
    df = pd.read_hdf(path)
    logger.info(f"Loaded traffic data: {df.shape} from {path}")
    return df


def load_adjacency(pkl_path: Path) -> Tuple[np.ndarray, list]:
    """Load the adjacency matrix pickle shipped with METR-LA / PeMS-BAY."""
    with open(pkl_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
    logger.info(f"Adjacency matrix loaded: {adj_mx.shape}, {len(sensor_ids)} sensors")
    return adj_mx, sensor_ids


# ── Sliding-window helper ─────────────────────────────────────────────────────

def create_windows(
    data: np.ndarray,     # (T, N, F)
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised sliding-window generation.

    Returns:
        X: (samples, seq_len, N, F)
        Y: (samples, horizon, N, F)
    """
    T = data.shape[0]
    xs, ys = [], []
    for t in range(T - seq_len - horizon + 1):
        xs.append(data[t : t + seq_len])
        ys.append(data[t + seq_len : t + seq_len + horizon])
    return np.stack(xs), np.stack(ys)


# ── Time encoding ─────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> np.ndarray:
    """Append cyclic hour-of-day and day-of-week encodings.

    Returns array of shape (T, N, F+4) — flow + sin/cos hour + sin/cos dow.
    """
    flow = df.values[:, :, None].astype(np.float32)  # (T, N, 1)
    T, N, _ = flow.shape

    tod = df.index.hour + df.index.minute / 60.0      # fraction of day
    dow = df.index.dayofweek.astype(np.float32)

    sin_h = np.sin(2 * np.pi * tod / 24.0).reshape(T, 1, 1) * np.ones((1, N, 1))
    cos_h = np.cos(2 * np.pi * tod / 24.0).reshape(T, 1, 1) * np.ones((1, N, 1))
    sin_d = np.sin(2 * np.pi * dow / 7.0).reshape(T, 1, 1) * np.ones((1, N, 1))
    cos_d = np.cos(2 * np.pi * dow / 7.0).reshape(T, 1, 1) * np.ones((1, N, 1))

    return np.concatenate([flow, sin_h, cos_h, sin_d, cos_d], axis=-1)  # (T, N, 5)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    """Window-sliced traffic dataset with optional normalisation."""

    def __init__(
        self,
        x: np.ndarray,   # (S, seq_len, N, F)
        y: np.ndarray,   # (S, horizon, N, F)
    ) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# ── High-level builder ────────────────────────────────────────────────────────

def build_dataloaders(
    data_cfg: Optional[DataConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, np.ndarray, list]:
    """End-to-end pipeline: raw HDF5 → normalised DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, scaler, adj_mx, sensor_ids
    """
    data_cfg = data_cfg or cfg.data
    meta      = DATASET_REGISTRY[data_cfg.dataset]

    # -- Load raw data ---------------------------------------------------------
    df = load_traffic_h5(meta["raw_h5"])
    adj_mx, sensor_ids = load_adjacency(meta["adj_pkl"])

    # -- Feature engineering --------------------------------------------------
    arr = add_time_features(df)                   # (T, N, 5)

    # -- Train / val / test split (chronological) -----------------------------
    T = arr.shape[0]
    n_train = int(T * data_cfg.train_ratio)
    n_val   = int(T * data_cfg.val_ratio)

    train_data = arr[:n_train]
    val_data   = arr[n_train : n_train + n_val]
    test_data  = arr[n_train + n_val :]

    # -- Normalise (fit only on train) ----------------------------------------
    scaler = StandardScaler().fit(train_data)
    if data_cfg.normalize:
        train_data = scaler.transform(train_data)
        val_data   = scaler.transform(val_data)
        test_data  = scaler.transform(test_data)

    # -- Create sliding windows -----------------------------------------------
    x_tr, y_tr = create_windows(train_data, data_cfg.seq_len, data_cfg.horizon)
    x_va, y_va = create_windows(val_data,   data_cfg.seq_len, data_cfg.horizon)
    x_te, y_te = create_windows(test_data,  data_cfg.seq_len, data_cfg.horizon)

    logger.info(
        f"Dataset splits — train: {len(x_tr)}, val: {len(x_va)}, test: {len(x_te)} samples"
    )

    # -- DataLoaders ----------------------------------------------------------
    def _loader(x, y, shuffle: bool) -> DataLoader:
        return DataLoader(
            TrafficDataset(x, y),
            batch_size=data_cfg.batch_size,
            shuffle=shuffle,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            drop_last=False,
        )

    return (
        _loader(x_tr, y_tr, shuffle=True),
        _loader(x_va, y_va, shuffle=False),
        _loader(x_te, y_te, shuffle=False),
        scaler,
        adj_mx,
        sensor_ids,
    )
