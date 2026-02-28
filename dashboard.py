"""
SmarTraffic real-time dashboard
================================

FastAPI application with:
  * REST endpoints for batch predictions & model info
  * WebSocket endpoint that pushes live predictions every N seconds
  * Static HTML/JS frontend served from templates/

Usage
-----
    python dashboard.py
    # → http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_DIR, DATA_DIR, cfg
from src.model import build_model
from src.utils import load_checkpoint, resolve_device

# ── Global state ──────────────────────────────────────────────────────────────
_model:  Optional[torch.nn.Module] = None
_device: Optional[torch.device]   = None
_adj:    Optional[np.ndarray]     = None
_sensor_ids: list[str]            = []


# ── Lifespan (model warm-up) ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device, _adj, _sensor_ids

    _device = resolve_device(cfg.train.device)

    # Load adjacency
    adj_path = DATA_DIR / cfg.data.dataset / "adj_mx.npy"
    if adj_path.exists():
        _adj = np.load(adj_path)
    else:
        logger.warning(f"Adjacency not found at {adj_path}, using random 207×207")
        _adj = np.random.rand(207, 207).astype(np.float32)
        np.fill_diagonal(_adj, 0)

    # Load sensor IDs
    coords_path = DATA_DIR / cfg.data.dataset / "node_coords.csv"
    if coords_path.exists():
        import pandas as pd
        df            = pd.read_csv(coords_path)
        _sensor_ids   = df["sensor_id"].astype(str).tolist()
    else:
        _sensor_ids = [str(i) for i in range(_adj.shape[0])]

    # Build and optionally load checkpoint
    _model = build_model(_adj).to(_device)
    best_ckpt = MODEL_DIR / "dcrnn_best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, _model, device=_device)
        logger.info("Loaded best checkpoint for serving")
    else:
        logger.warning("No checkpoint found — serving with untrained model (random weights)")

    _model.eval()
    logger.info("Dashboard startup complete")
    yield
    logger.info("Dashboard shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmarTraffic API",
    description="Real-time traffic flow prediction powered by DCRNN",
    version="0.1.0",
    lifespan=lifespan,
)

# Serve static files if directory exists
static_dir = ROOT_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Input features for a single prediction request.
    `x` has shape (seq_len, num_nodes, num_features).
    """
    x: list[list[list[float]]]
    horizon: int = 12


class PredictionResponse(BaseModel):
    sensor_ids:    list[str]
    horizon_steps: int
    predictions:   list[list[float]]  # (H, N)
    timestamp:     float


class ModelInfo(BaseModel):
    num_nodes:      int
    num_parameters: int
    input_dim:      int
    output_dim:     int
    rnn_units:      int
    horizon:        int
    dataset:        str
    checkpoint_exists: bool


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard page."""
    html_path = ROOT_DIR / "templates" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse(content=_fallback_html(), status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return ModelInfo(
        num_nodes          = _adj.shape[0],
        num_parameters     = _model.count_parameters(),
        input_dim          = cfg.model.input_dim,
        output_dim         = cfg.model.output_dim,
        rnn_units          = cfg.model.rnn_units,
        horizon            = cfg.data.horizon,
        dataset            = cfg.data.dataset,
        checkpoint_exists  = (MODEL_DIR / "dcrnn_best.pt").exists(),
    )


@app.post("/traffic/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """
    Run a single traffic flow prediction.

    Body:
        x: list of shape (seq_len, num_nodes, num_features)
        horizon: optional override (ignored — uses model config)
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    N = _adj.shape[0]
    try:
        x_np = np.array(req.x, dtype=np.float32)        # (T, N, F)
        if x_np.ndim != 3 or x_np.shape[1] != N:
            raise ValueError(f"Expected x shape (T, {N}, F), got {x_np.shape}")
    except Exception as exc:
        raise HTTPException(422, f"Invalid input: {exc}")

    x_t = torch.from_numpy(x_np).unsqueeze(0).to(_device)  # (1, T, N, F)

    with torch.no_grad():
        pred = _model.predict(x_t)                           # (1, H, N, 1)

    predictions = pred[0, :, :, 0].cpu().numpy().tolist()   # (H, N)

    return PredictionResponse(
        sensor_ids    = _sensor_ids,
        horizon_steps = len(predictions),
        predictions   = predictions,
        timestamp     = time.time(),
    )


@app.get("/traffic/demo")
async def demo_prediction():
    """Generate a demo prediction from random input (for testing without real data)."""
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    N   = _adj.shape[0]
    T   = cfg.data.seq_len
    F   = cfg.model.input_dim
    x_t = torch.randn(1, T, N, F, device=_device)

    with torch.no_grad():
        pred = _model.predict(x_t)

    predictions = pred[0, :, :, 0].cpu().numpy().tolist()
    return PredictionResponse(
        sensor_ids    = _sensor_ids,
        horizon_steps = len(predictions),
        predictions   = predictions,
        timestamp     = time.time(),
    )


# ── WebSocket (live streaming) ────────────────────────────────────────────────

class ConnectionManager:
    """Track active WebSocket connections."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS connected — {len(self.active)} clients")

    def disconnect(self, ws: WebSocket) -> None:
        self.active.remove(ws)
        logger.info(f"WS disconnected — {len(self.active)} clients")

    async def broadcast(self, data: Any) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


@app.websocket("/ws/predictions")
async def ws_predictions(ws: WebSocket):
    """
    WebSocket endpoint.
    Sends a new prediction payload every `ws_interval_s` seconds.
    """
    await manager.connect(ws)
    try:
        while True:
            N   = _adj.shape[0]
            T   = cfg.data.seq_len
            F   = cfg.model.input_dim
            x_t = torch.randn(1, T, N, F, device=_device)

            with torch.no_grad():
                pred = _model.predict(x_t)

            # Return mean predicted flow per node at horizon step 0 and last step
            flow_now  = pred[0, 0, :, 0].cpu().numpy().tolist()
            flow_last = pred[0, -1, :, 0].cpu().numpy().tolist()

            payload = {
                "timestamp":   time.time(),
                "sensor_ids":  _sensor_ids[:50],  # cap for demo
                "flow_t1":     flow_now[:50],
                "flow_tH":     flow_last[:50],
            }
            await ws.send_json(payload)
            await asyncio.sleep(cfg.dashboard.ws_interval_s)

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Fallback HTML (no templates dir) ─────────────────────────────────────────

def _fallback_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SmarTraffic Dashboard</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 2rem; }
    h1   { color: #58a6ff; }
    .card { background: #161b22; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
    code { background: #21262d; padding: 2px 6px; border-radius: 4px; }
    #log { height: 300px; overflow-y: auto; background: #010409; padding: 1rem; border-radius: 8px; font-family: monospace; font-size: 0.85rem; }
  </style>
</head>
<body>
  <h1>🚦 SmarTraffic Dashboard</h1>
  <div class="card">
    <h2>Live Prediction Feed</h2>
    <p>WebSocket: <code>ws://localhost:8000/ws/predictions</code></p>
    <div id="log">Connecting…</div>
  </div>
  <div class="card">
    <h2>REST API</h2>
    <p><a href="/docs" style="color:#58a6ff">Interactive API Docs →</a></p>
  </div>
  <script>
    const log = document.getElementById('log');
    const ws  = new WebSocket('ws://' + location.host + '/ws/predictions');
    ws.onopen  = () => { log.innerHTML = '<span style="color:#3fb950">Connected ✓</span><br>'; };
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      const line = `[${new Date(d.timestamp * 1000).toLocaleTimeString()}] `
                 + `Sensors: ${d.sensor_ids.length} | `
                 + `Flow T+1 avg: ${(d.flow_t1.reduce((a,b)=>a+b,0)/d.flow_t1.length).toFixed(2)} | `
                 + `Flow T+H avg: ${(d.flow_tH.reduce((a,b)=>a+b,0)/d.flow_tH.length).toFixed(2)}`;
      log.innerHTML += line + '<br>';
      log.scrollTop = log.scrollHeight;
    };
    ws.onerror  = (e) => { log.innerHTML += '<span style="color:#f85149">Error</span><br>'; };
    ws.onclose  = ()  => { log.innerHTML += '<span style="color:#e3b341">Disconnected</span><br>'; };
  </script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "dashboard:app",
        host=cfg.dashboard.host,
        port=cfg.dashboard.port,
        reload=cfg.dashboard.reload,
        log_level="info",
    )
