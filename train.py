"""
Training entry point for the SmarTraffic DCRNN model.

    python train.py --dataset metr-la --horizon 12 --epochs 100

Features
--------
* Automatic mixed precision (torch.cuda.amp)
* Cosine-annealed or milestone learning-rate schedule
* Early stopping on validation MAE
* Curriculum learning with exponential teacher-forcing decay
* Rich progress bars and console metrics via `rich`
* TensorBoard-compatible CSV logging to logs/
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.cuda.amp import GradScaler, autocast

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_DIR, LOG_DIR, cfg
from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import (
    compute_all_metrics,
    load_checkpoint,
    masked_mae,
    resolve_device,
    save_checkpoint,
    seed_everything,
)

console = Console()


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DCRNN on traffic data")
    p.add_argument("--dataset",  choices=["metr-la", "pems-bay"], default="metr-la")
    p.add_argument("--horizon",  type=int, default=12, help="prediction horizon (steps)")
    p.add_argument("--epochs",   type=int, default=None)
    p.add_argument("--lr",       type=float, default=None)
    p.add_argument("--batch",    type=int, default=None)
    p.add_argument("--device",   type=str, default="auto")
    p.add_argument("--resume",   type=str, default=None, help="path to checkpoint to resume")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--no-amp",   action="store_true", help="disable automatic mixed precision")
    return p.parse_args()


# ── CSV logger ────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f   = open(path, "w", newline="")
        self._w   = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._w.writeheader()

    def log(self, row: dict) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ── Train / evaluate loops ────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    use_amp:   bool,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Train"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("train", total=n_batches)

        for x, y in loader:
            x = x.to(device, non_blocking=True)   # (B, T, N, F)
            y = y.to(device, non_blocking=True)   # (B, H, N, F)

            # Use only the first feature (flow) as target
            y_target = y[..., :1]                 # (B, H, N, 1)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                pred = model(x, targets=y_target)  # (B, H, N, output_dim)
                loss = masked_mae(pred, y_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            prog.advance(task)

    return {"loss": total_loss / n_batches}


@torch.no_grad()
def evaluate(
    model:   nn.Module,
    loader,
    device:  torch.device,
    scaler_obj,   # StandardScaler for denorm
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    all_preds, all_true = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_target = y[..., :1]

        with autocast(enabled=use_amp):
            pred = model(x)

        # Denormalise for real-world metrics
        pred_np = scaler_obj.inverse_transform(pred.cpu())
        true_np = scaler_obj.inverse_transform(y_target.cpu())
        all_preds.append(pred_np)
        all_true.append(true_np)

    preds = torch.cat(all_preds, dim=0)
    trues = torch.cat(all_true, dim=0)
    return compute_all_metrics(preds, trues)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # -- Apply CLI overrides to cfg --
    cfg.data.dataset  = args.dataset
    cfg.data.horizon  = args.horizon
    if args.epochs: cfg.train.epochs     = args.epochs
    if args.lr:     cfg.train.lr         = args.lr
    if args.batch:  cfg.data.batch_size  = args.batch
    cfg.train.device = args.device
    cfg.train.amp    = not args.no_amp

    seed_everything(args.seed)
    device  = resolve_device(cfg.train.device)
    use_amp = cfg.train.amp and device.type == "cuda"

    # -- Data ------------------------------------------------------------------
    logger.info("Building data loaders …")
    train_loader, val_loader, test_loader, data_scaler, adj_mx, _ = build_dataloaders(cfg.data)

    # -- Model -----------------------------------------------------------------
    model = build_model(adj_mx).to(device)
    console.print(
        f"[green]Model:[/green] DCRNN — "
        f"[bold]{model.count_parameters():,}[/bold] trainable parameters"
    )

    # -- Optimiser & scheduler -------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.train.lr_milestones,
        gamma=cfg.train.lr_decay_ratio,
    )
    grad_scaler = GradScaler(enabled=use_amp)

    # -- Resume ----------------------------------------------------------------
    start_epoch = 0
    best_mae    = float("inf")
    patience    = 0
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        best_mae    = ckpt["metrics"].get("val_MAE", best_mae)

    # -- Logging ---------------------------------------------------------------
    run_id  = time.strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_DIR / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_log = CSVLogger(
        log_dir / "metrics.csv",
        ["epoch", "train_loss", "val_MAE", "val_RMSE", "val_MAPE", "lr"],
    )
    logger.add(log_dir / "train.log", level="INFO")

    # -- Training loop ---------------------------------------------------------
    for epoch in range(start_epoch, cfg.train.epochs):
        console.rule(f"[cyan]Epoch {epoch + 1}/{cfg.train.epochs}")

        t0         = time.time()
        train_info = train_epoch(
            model, train_loader, optimizer, grad_scaler, device, use_amp, cfg.train.grad_clip
        )
        scheduler.step()

        if (epoch + 1) % cfg.train.val_every == 0:
            val_metrics = evaluate(model, val_loader, device, data_scaler, use_amp)

            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]

            # -- Pretty table --
            tbl = Table(show_header=True, header_style="bold magenta")
            tbl.add_column("Metric")
            tbl.add_column("Value", justify="right")
            tbl.add_row("Train Loss",   f"{train_info['loss']:.4f}")
            tbl.add_row("Val MAE",       f"{val_metrics['MAE']:.4f}")
            tbl.add_row("Val RMSE",      f"{val_metrics['RMSE']:.4f}")
            tbl.add_row("Val MAPE (%)",  f"{val_metrics['MAPE']:.2f}")
            tbl.add_row("LR",            f"{lr_now:.2e}")
            tbl.add_row("Time (s)",      f"{elapsed:.1f}")
            console.print(tbl)

            csv_log.log({
                "epoch":      epoch + 1,
                "train_loss": round(train_info["loss"], 6),
                "val_MAE":    round(val_metrics["MAE"],  6),
                "val_RMSE":   round(val_metrics["RMSE"], 6),
                "val_MAPE":   round(val_metrics["MAPE"], 4),
                "lr":         lr_now,
            })

            # -- Checkpoint --
            is_best = val_metrics["MAE"] < best_mae
            if is_best:
                best_mae = val_metrics["MAE"]
                patience = 0
                save_checkpoint(
                    MODEL_DIR / "dcrnn_best.pt",
                    model, optimizer, epoch,
                    {"val_MAE": best_mae},
                    cfg.to_dict(),
                )
            else:
                patience += 1
                if patience >= cfg.train.patience:
                    console.print(
                        f"[yellow]Early stopping triggered after {patience} epochs "
                        f"without improvement.[/yellow]"
                    )
                    break

    # -- Final test evaluation -------------------------------------------------
    console.rule("[green]Test Evaluation")
    best_ckpt = MODEL_DIR / "dcrnn_best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, device=device)
    test_metrics = evaluate(model, test_loader, device, data_scaler, use_amp)

    tbl = Table(title="Test Results", header_style="bold green")
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    for k, v in test_metrics.items():
        tbl.add_row(k, f"{v:.4f}")
    console.print(tbl)

    csv_log.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
