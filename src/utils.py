"""
Utility functions for SmarTraffic.

Covers:
  - Reproducibility seeding
  - Graph diffusion matrix construction
  - Z-score normalisation / denormalisation
  - Traffic metrics (MAE, RMSE, MAPE)
  - Checkpoint save / load
  - Device resolution
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from loguru import logger


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Global seed set to {seed}")


# ── Device ─────────────────────────────────────────────────────────────────────

def resolve_device(device: str = "auto") -> torch.device:
    """Return the best available torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            d = torch.device("cuda")
        elif torch.backends.mps.is_available():
            d = torch.device("mps")
        else:
            d = torch.device("cpu")
    else:
        d = torch.device(device)
    logger.info(f"Using device: {d}")
    return d


# ── Sparse / adjacency helpers ────────────────────────────────────────────────

def load_adj_from_pickle(pkl_path: Path) -> Tuple[list, int]:
    """Load adjacency matrix from a METR-LA / PeMS-BAY pickle file.

    Returns:
        adj_mx_list: list of np.ndarray adjacency matrices
        num_nodes: int
    """
    with open(pkl_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
    return adj_mx, len(sensor_ids)


def distance_to_weight(distances: np.ndarray, sigma2: float = 0.1, epsilon: float = 0.5) -> np.ndarray:
    """Convert a distance matrix to a Gaussian kernel adjacency matrix.

    W_{ij} = exp(−d²_{ij} / σ²)  if W_{ij} ≥ ε, else 0

    Args:
        distances: (N, N) distance matrix (e.g. road distances in metres)
        sigma2:    kernel bandwidth
        epsilon:   sparsity threshold

    Returns:
        W: (N, N) weighted adjacency matrix
    """
    n = distances.shape[0]
    W = np.zeros_like(distances, dtype=np.float32)
    std = distances[distances != np.inf].std()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = distances[i, j]
            if d == np.inf:
                continue
            w = np.exp(-d ** 2 / (std ** 2 * sigma2))
            if w >= epsilon:
                W[i, j] = w
    return W


def compute_diffusion_matrices(
    adj: np.ndarray,
    filter_type: str = "dual_random_walk",
) -> list[np.ndarray]:
    """Compute transition matrices for diffusion convolution.

    Args:
        adj:         (N, N) adjacency matrix (can be asymmetric)
        filter_type: one of {"dual_random_walk", "random_walk", "laplacian"}

    Returns:
        List of sparse 2-D arrays; will be used as graph filters.
    """
    def _row_normalize(mx: np.ndarray) -> sp.csr_matrix:
        row_sums = np.array(mx.sum(1)).flatten()
        row_sums[row_sums == 0] = 1.0          # avoid div-by-zero
        d_inv = 1.0 / row_sums
        d_mat = sp.diags(d_inv)
        return d_mat.dot(sp.csr_matrix(mx))

    if filter_type == "laplacian":
        d = np.diag(adj.sum(1))
        lap = sp.eye(adj.shape[0]) - _row_normalize(adj)
        return [sparse_to_numpy(lap)]

    if filter_type == "random_walk":
        return [sparse_to_numpy(_row_normalize(adj))]

    # dual_random_walk: forward + backward
    return [
        sparse_to_numpy(_row_normalize(adj)),
        sparse_to_numpy(_row_normalize(adj.T)),
    ]


def sparse_to_numpy(mx: sp.spmatrix | np.ndarray) -> np.ndarray:
    if sp.issparse(mx):
        return np.array(mx.todense(), dtype=np.float32)
    return np.asarray(mx, dtype=np.float32)


def build_support(adj: np.ndarray, filter_type: str, max_diffusion_step: int) -> list[np.ndarray]:
    """
    Build Chebyshev / random-walk polynomial support matrices up to order K.

    Returns a flat list of (N, N) numpy arrays.
    """
    supports = compute_diffusion_matrices(adj, filter_type)
    n = adj.shape[0]
    results: list[np.ndarray] = [np.eye(n, dtype=np.float32)]   # K=0: identity

    for base in supports:
        pk = np.eye(n, dtype=np.float32)
        for _ in range(max_diffusion_step):
            pk = pk @ base
            results.append(pk.copy())

    return results


# ── Normalisation ─────────────────────────────────────────────────────────────

class StandardScaler:
    """Z-score normaliser fitted on training data per feature."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """x shape: (T, N, F)"""
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std  = x.std(axis=(0, 1), keepdims=True)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
            std  = torch.tensor(self.std,  dtype=x.dtype, device=x.device)
            return x * std + mean
        return x * self.std + self.mean

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        data = np.load(path)
        scaler = cls()
        scaler.mean = data["mean"]
        scaler.std  = data["std"]
        return scaler


# ── Metrics ───────────────────────────────────────────────────────────────────

def masked_mae(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Mean Absolute Error, masking zero ground-truth values (missing sensors)."""
    mask = true != null_val
    loss = torch.abs(pred[mask] - true[mask])
    return loss.mean()


def masked_rmse(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = true != null_val
    loss = (pred[mask] - true[mask]) ** 2
    return torch.sqrt(loss.mean())


def masked_mape(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Mean Absolute Percentage Error (returns %, not fraction)."""
    mask = (true != null_val) & (true.abs() > 1e-5)
    loss = torch.abs((pred[mask] - true[mask]) / true[mask])
    return loss.mean() * 100.0


def compute_all_metrics(
    pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0
) -> dict[str, float]:
    return {
        "MAE":  masked_mae(pred, true, null_val).item(),
        "RMSE": masked_rmse(pred, true, null_val).item(),
        "MAPE": masked_mape(pred, true, null_val).item(),
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg_dict: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg_dict,
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"Checkpoint loaded ← {path}  (epoch {ckpt['epoch']})")
    return ckpt


# ── Curriculum learning ───────────────────────────────────────────────────────

def curriculum_tf_prob(global_step: int, cl_decay_steps: int) -> float:
    """Exponentially decaying teacher-forcing probability for curriculum learning."""
    return max(0.0, 1.0 - global_step / cl_decay_steps)
