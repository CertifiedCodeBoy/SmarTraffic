"""
Download METR-LA / PeMS-BAY datasets from Google Drive.

    python data/download_data.py --dataset metr-la
    python data/download_data.py --dataset pems-bay

Files written (matching DATASET_REGISTRY paths):
    data/metr-la/metr-la.h5
    data/metr-la/adj_mx.pkl
    data/metr-la/graph_sensor_locations.csv
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET_REGISTRY, DATA_DIR
from src.utils import logger


SENSOR_LOCS_URL = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/"
    "sensor_graph/graph_sensor_locations.csv"
)


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Stream-download `url` to `dest`, showing a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"Already downloaded: {dest}")
        return dest

    logger.info(f"Downloading {url} → {dest}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def download_metr_la() -> None:
    meta    = DATASET_REGISTRY["metr-la"]
    out_dir = DATA_DIR / "metr-la"
    out_dir.mkdir(parents=True, exist_ok=True)

    download_file(meta["url"],     meta["raw_h5"])
    download_file(meta["adj_url"], meta["adj_pkl"])
    download_file(SENSOR_LOCS_URL, out_dir / "graph_sensor_locations.csv")

    logger.info("METR-LA download complete.")


def download_pems_bay() -> None:
    meta    = DATASET_REGISTRY["pems-bay"]
    out_dir = DATA_DIR / "pems-bay"
    out_dir.mkdir(parents=True, exist_ok=True)

    download_file(meta["url"],     meta["raw_h5"])
    download_file(meta["adj_url"], meta["adj_pkl"])

    logger.info("PeMS-BAY download complete.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download traffic datasets")
    p.add_argument("--dataset", choices=["metr-la", "pems-bay", "all"], default="metr-la")
    args = p.parse_args()

    if args.dataset in ("metr-la", "all"):
        download_metr_la()
    if args.dataset in ("pems-bay", "all"):
        download_pems_bay()


if __name__ == "__main__":
    main()
