"""
Road-network graph construction.

Usage
-----
    # METR-LA (uses shipped adjacency pickle)
    python src/build_graph.py --city metr-la

    # Algerian urban grid (synthetic, random-seed based)
    python src/build_graph.py --city algiers --nodes 100

The script writes the following artefacts to data/<city>/:
    adj_mx.npy           – dense adjacency matrix  (N × N)
    node_coords.csv      – lat, lon, sensor_id
    graph.gpickle        – NetworkX DiGraph
    graph_map.html       – interactive Folium map
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config import DATA_DIR, DATASET_REGISTRY, cfg
from src.utils import distance_to_weight, logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def coords_to_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute pairwise Haversine distance matrix (N × N) in metres."""
    N = len(lats)
    D = np.full((N, N), np.inf, dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = haversine(lats[i], lons[i], lats[j], lons[j])
    return D


def adj_to_networkx(adj: np.ndarray, coords: pd.DataFrame) -> nx.DiGraph:
    """Convert adjacency matrix + coords to a NetworkX DiGraph."""
    G = nx.DiGraph()
    for idx, row in coords.iterrows():
        G.add_node(idx, lat=row["lat"], lon=row["lon"], sensor_id=row["sensor_id"])
    rows, cols = np.where(adj > 0)
    for r, c in zip(rows, cols):
        G.add_edge(int(r), int(c), weight=float(adj[r, c]))
    return G


def save_folium_map(
    coords: pd.DataFrame,
    G: nx.DiGraph,
    out_path: Path,
    center: tuple[float, float] = (34.0195, -118.4912),
    zoom: int = 11,
) -> None:
    """Render graph as interactive Folium map with edges and sensor markers."""
    try:
        import folium
    except ImportError:
        logger.warning("folium not installed – skipping map generation")
        return

    m = folium.Map(location=list(center), zoom_start=zoom, tiles="CartoDB positron")

    # Draw edges (sample for performance)
    edge_list = list(G.edges(data=True))[:2000]
    for u, v, data in edge_list:
        lat_u, lon_u = coords.loc[u, ["lat", "lon"]]
        lat_v, lon_v = coords.loc[v, ["lat", "lon"]]
        folium.PolyLine(
            [(lat_u, lon_u), (lat_v, lon_v)],
            weight=1.5,
            color="#3388ff",
            opacity=0.6,
        ).add_to(m)

    # Draw sensor nodes
    for idx, row in coords.iterrows():
        folium.CircleMarker(
            location=(row["lat"], row["lon"]),
            radius=4,
            color="#e84040",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Sensor {row['sensor_id']} (node {idx})",
        ).add_to(m)

    m.save(str(out_path))
    logger.info(f"Interactive map saved → {out_path}")


# ── METR-LA builder ───────────────────────────────────────────────────────────

def build_metr_la(out_dir: Path) -> None:
    """Load METR-LA adjacency pickle and derive GeoDataFrame + NetworkX graph."""
    meta    = DATASET_REGISTRY["metr-la"]
    adj_pkl = meta["adj_pkl"]

    if not adj_pkl.exists():
        logger.error(
            f"Adjacency pickle not found at {adj_pkl}.\n"
            "Run `python data/download_data.py --dataset metr-la` first."
        )
        return

    with open(adj_pkl, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

    adj_mx = adj_mx.astype(np.float32)
    np.save(out_dir / "adj_mx.npy", adj_mx)

    # METR-LA ships a separate sensor coordinates file (locations.csv)
    locs_path = out_dir / "graph_sensor_locations.csv"
    if locs_path.exists():
        locs = pd.read_csv(locs_path, index_col=0)
        locs.columns = [c.strip() for c in locs.columns]
        coords = pd.DataFrame({
            "sensor_id": sensor_ids,
            "lat":        locs["latitude"].values,
            "lon":        locs["longitude"].values,
        })
    else:
        logger.warning("Sensor locations CSV not found – using placeholder coords")
        N = len(sensor_ids)
        coords = pd.DataFrame({
            "sensor_id": sensor_ids,
            "lat":        np.random.uniform(33.9, 34.1, N),
            "lon":        np.random.uniform(-118.6, -118.1, N),
        })

    coords.to_csv(out_dir / "node_coords.csv", index=True)

    G = adj_to_networkx(adj_mx, coords)
    nx.write_gpickle(G, out_dir / "graph.gpickle")

    save_folium_map(
        coords, G, out_dir / "graph_map.html",
        center=cfg.dashboard.map_center, zoom=cfg.dashboard.map_zoom,
    )
    logger.info(
        f"METR-LA graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )


# ── Algiers synthetic builder ─────────────────────────────────────────────────

def build_algiers(out_dir: Path, num_nodes: int = 100, seed: int = 42) -> None:
    """Generate a synthetic urban grid for Algiers.

    Uses a Waxman random-graph model (nodes ~ GPS coords in Algiers bounding box)
    with distance-based edge weights.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Algiers bounding box (approx.)
    lat_min, lat_max = 36.70, 36.82
    lon_min, lon_max =  2.95,  3.16

    lats = rng.uniform(lat_min, lat_max, num_nodes)
    lons = rng.uniform(lon_min, lon_max, num_nodes)

    D = coords_to_distance_matrix(lats, lons)
    adj_mx = distance_to_weight(D, sigma2=cfg.graph.sigma2, epsilon=cfg.graph.epsilon)

    np.save(out_dir / "adj_mx.npy", adj_mx)

    coords = pd.DataFrame({
        "sensor_id": [f"ALG_{i:03d}" for i in range(num_nodes)],
        "lat":        lats,
        "lon":        lons,
    })
    coords.to_csv(out_dir / "node_coords.csv", index=True)

    G = adj_to_networkx(adj_mx, coords)
    nx.write_gpickle(G, out_dir / "graph.gpickle")

    save_folium_map(
        coords, G, out_dir / "graph_map.html",
        center=(36.752, 3.042), zoom=12,
    )
    logger.info(
        f"Algiers synthetic graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
        f"Adjacency density: {adj_mx.astype(bool).mean():.3%}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build road-network graph")
    p.add_argument("--city",  choices=["metr-la", "pems-bay", "algiers"], default="metr-la")
    p.add_argument("--nodes", type=int, default=100, help="nodes for synthetic city")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.city == "metr-la":
        out_dir = DATA_DIR / "metr-la"
        out_dir.mkdir(parents=True, exist_ok=True)
        build_metr_la(out_dir)

    elif args.city == "pems-bay":
        out_dir = DATA_DIR / "pems-bay"
        out_dir.mkdir(parents=True, exist_ok=True)
        build_metr_la(out_dir)   # same pickle format

    elif args.city == "algiers":
        out_dir = DATA_DIR / "city_graph"
        build_algiers(out_dir, num_nodes=args.nodes, seed=args.seed)

    else:
        logger.error(f"Unknown city: {args.city}")
        sys.exit(1)


if __name__ == "__main__":
    main()
