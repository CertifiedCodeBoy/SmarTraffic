"""
DCRNN — Diffusion Convolutional Recurrent Neural Network
=========================================================

Paper: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
        Forecasting" (Li et al., ICLR 2018)  https://arxiv.org/abs/1707.01926

Architecture
------------
                 ┌─────────────────────────────────────────┐
  Input          │  Diffusion Conv  →  DCGRUCell  × L      │  Encoder
  (B,T,N,F)  ──► │  (spatial filter)   (temporal)          │
                 └────────────────────┬────────────────────┘
                                      │ hidden states
                 ┌────────────────────▼────────────────────┐
                 │  DCGRUCell × L  →  FC projection        │  Decoder
                 │  (autoregressive, curriculum learning)   │
                 └─────────────────────────────────────────┘
                                      │
                              (B, H, N, output_dim)

Modules
-------
  DiffusionConvolution   – graph convolution with diffusion support
  DCGRUCell              – GRU cell with graph conv replacing linear ops
  DCRNNEncoder           – stacked encoder
  DCRNNDecoder           – stacked autoregressive decoder
  DCRNNModel             – full seq2seq wrapper
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Diffusion Convolution ─────────────────────────────────────────────────────

class DiffusionConvolution(nn.Module):
    """
    Graph convolution via polynomial approximation of the graph diffusion
    operator (random walk / Laplacian), as in Eq.(3) of Li et al. 2018.

    Z = Σ_k Σ_p ( θ_{k,p}   ·  T_p(D^{-1} A)  X )
              support matrices pre-computed, passed as `support`

    Args:
        in_features:  number of input features C_in
        out_features: number of output features C_out
        num_supports: len(support) — 2 for dual random walk, 1 otherwise
    """

    def __init__(self, in_features: int, out_features: int, num_supports: int) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.num_supports = num_supports

        # Weight tensor: one matrix per support polynomial
        # shape: (num_supports, in_features, out_features)
        self.weight = nn.Parameter(
            torch.empty(num_supports, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,                # (B, N, C_in)
        supports: List[torch.Tensor],   # list of (N, N) tensors
    ) -> torch.Tensor:                  # (B, N, C_out)
        B, N, _ = x.shape
        outputs = []
        for i, S in enumerate(supports):
            # S: (N, N)   x: (B, N, C_in)
            hx = torch.einsum("nm,bmc->bnc", S, x)     # (B, N, C_in)
            hx = torch.einsum("bnc,co->bno", hx, self.weight[i])  # (B, N, C_out)
            outputs.append(hx)
        out = sum(outputs) + self.bias
        return out   # (B, N, C_out)


# ── DCGRUCell ─────────────────────────────────────────────────────────────────

class DCGRUCell(nn.Module):
    """
    A GRU cell where the standard linear (fully connected) transformations
    on both input  and  hidden state  are replaced by graph diffusion convolutions.

    All gating equations follow the standard GRU formulation but using
    DiffusionConvolution instead of nn.Linear.
    """

    def __init__(
        self,
        input_dim: int,
        units: int,
        max_diffusion_step: int,
        num_nodes: int,
        filter_type: str = "dual_random_walk",
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.units      = units
        self.num_nodes  = num_nodes
        self.activation = torch.tanh if activation == "tanh" else torch.relu

        num_supports = 2 if filter_type == "dual_random_walk" else 1
        self._num_supports = num_supports

        # For each support we have (K+1) polynomials; +1 for the identity
        num_matrices = (max_diffusion_step + 1) * num_supports + 1

        # Gates: reset (r), update (z)  — combined
        self.conv_gate = DiffusionConvolution(
            in_features=input_dim + units,
            out_features=2 * units,
            num_supports=num_matrices,
        )
        # Candidate hidden state
        self.conv_cand = DiffusionConvolution(
            in_features=input_dim + units,
            out_features=units,
            num_supports=num_matrices,
        )

    def forward(
        self,
        x: torch.Tensor,                # (B, N, input_dim)
        h: torch.Tensor,                # (B, N, units)
        supports: List[torch.Tensor],   # pre-computed diff matrices
    ) -> torch.Tensor:                  # new h: (B, N, units)
        xh = torch.cat([x, h], dim=-1)          # (B, N, input_dim + units)
        gates = torch.sigmoid(self.conv_gate(xh, supports))  # (B, N, 2*units)
        r, z  = gates.chunk(2, dim=-1)           # each (B, N, units)

        xrh = torch.cat([x, r * h], dim=-1)     # (B, N, input_dim + units)
        c   = self.activation(self.conv_cand(xrh, supports))  # (B, N, units)

        h_new = z * h + (1 - z) * c
        return h_new


# ── Encoder ───────────────────────────────────────────────────────────────────

class DCRNNEncoder(nn.Module):
    """Stack of L DCGRUCells unrolled over the input sequence."""

    def __init__(
        self,
        input_dim:          int,
        units:              int,
        num_layers:         int,
        max_diffusion_step: int,
        num_nodes:          int,
        filter_type:        str = "dual_random_walk",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.units      = units
        self.num_nodes  = num_nodes

        self.cells = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.cells.append(
                DCGRUCell(in_dim, units, max_diffusion_step, num_nodes, filter_type)
            )
            in_dim = units

    def forward(
        self,
        x: torch.Tensor,                # (B, T, N, input_dim)
        supports: List[torch.Tensor],
    ) -> List[torch.Tensor]:            # list of L hidden states (B, N, units)
        B, T, N, _ = x.shape

        # Initialise hidden states
        hidden = [
            torch.zeros(B, N, self.units, device=x.device) for _ in range(self.num_layers)
        ]

        for t in range(T):
            inp = x[:, t, :, :]          # (B, N, input_dim)
            for l, cell in enumerate(self.cells):
                hidden[l] = cell(inp, hidden[l], supports)
                inp = hidden[l]

        return hidden


# ── Decoder ───────────────────────────────────────────────────────────────────

class DCRNNDecoder(nn.Module):
    """Autoregressive decoder with optional curriculum learning."""

    def __init__(
        self,
        output_dim:         int,
        units:              int,
        num_layers:         int,
        max_diffusion_step: int,
        num_nodes:          int,
        horizon:            int,
        filter_type:        str = "dual_random_walk",
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.units      = units
        self.num_layers = num_layers
        self.horizon    = horizon

        self.cells = nn.ModuleList()
        for _ in range(num_layers):
            self.cells.append(
                DCGRUCell(output_dim, units, max_diffusion_step, num_nodes, filter_type)
            )

        self.projection = nn.Linear(units, output_dim)

    def forward(
        self,
        hidden:    List[torch.Tensor],  # encoder final states
        supports:  List[torch.Tensor],
        targets:   Optional[torch.Tensor] = None,  # (B, H, N, output_dim)  for teacher forcing
        tf_prob:   float = 0.0,
    ) -> torch.Tensor:                 # (B, H, N, output_dim)
        B, N, _ = hidden[0].shape

        go_symbol = torch.zeros(B, N, self.output_dim, device=hidden[0].device)
        inp       = go_symbol
        h_states  = list(hidden)        # mutable copy
        outputs   = []

        for t in range(self.horizon):
            for l, cell in enumerate(self.cells):
                h_states[l] = cell(inp, h_states[l], supports)
                inp = h_states[l]      # pass to next layer

            pred = self.projection(inp)  # (B, N, output_dim)
            outputs.append(pred)

            # Curriculum learning: teacher forcing with scheduled probability
            if targets is not None and torch.rand(1).item() < tf_prob:
                inp = targets[:, t, :, :]   # (B, N, output_dim)
            else:
                inp = pred

        return torch.stack(outputs, dim=1)   # (B, H, N, output_dim)


# ── Full DCRNN model ──────────────────────────────────────────────────────────

class DCRNNModel(nn.Module):
    """
    End-to-end Diffusion Convolutional Recurrent Neural Network.

    Args:
        adj:                (N, N) adjacency matrix as numpy array
        input_dim:          number of input features per node per step
        output_dim:         number of predicted features per node per step
        seq_len:            encoder input sequence length
        horizon:            prediction horizon
        rnn_units:          GRU hidden size
        num_rnn_layers:     number of stacked GRU layers
        max_diffusion_step: polynomial order K
        filter_type:        diffusion type
    """

    def __init__(
        self,
        adj:                np.ndarray,
        input_dim:          int,
        output_dim:         int,
        seq_len:            int,
        horizon:            int,
        rnn_units:          int   = 64,
        num_rnn_layers:     int   = 2,
        max_diffusion_step: int   = 2,
        filter_type:        str   = "dual_random_walk",
        cl_decay_steps:     int   = 2000,
    ) -> None:
        super().__init__()

        from src.utils import build_support

        self.cl_decay_steps = cl_decay_steps
        self._global_step   = 0

        # Pre-compute & register graph support matrices as buffers
        support_ndarrays = build_support(adj, filter_type, max_diffusion_step)
        for i, s in enumerate(support_ndarrays):
            self.register_buffer(f"support_{i}", torch.from_numpy(s).float())
        self._num_supports = len(support_ndarrays)

        self.encoder = DCRNNEncoder(
            input_dim, rnn_units, num_rnn_layers, max_diffusion_step,
            adj.shape[0], filter_type,
        )
        self.decoder = DCRNNDecoder(
            output_dim, rnn_units, num_rnn_layers, max_diffusion_step,
            adj.shape[0], horizon, filter_type,
        )

    @property
    def supports(self) -> List[torch.Tensor]:
        return [getattr(self, f"support_{i}") for i in range(self._num_supports)]

    def _tf_prob(self) -> float:
        if not self.training:
            return 0.0
        return max(0.0, 1.0 - self._global_step / self.cl_decay_steps)

    def forward(
        self,
        x:       torch.Tensor,                    # (B, T, N, F)
        targets: Optional[torch.Tensor] = None,   # (B, H, N, output_dim)
    ) -> torch.Tensor:                             # (B, H, N, output_dim)
        hidden  = self.encoder(x, self.supports)
        output  = self.decoder(hidden, self.supports, targets, tf_prob=self._tf_prob())
        if self.training:
            self._global_step += 1
        return output

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Inference-only forward (no teacher forcing, no grad)."""
        self.eval()
        return self(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(adj: np.ndarray) -> DCRNNModel:
    """Instantiate a DCRNNModel from the global config singleton."""
    from config import cfg
    m = cfg.model
    d = cfg.data
    return DCRNNModel(
        adj               = adj,
        input_dim         = m.input_dim,
        output_dim        = m.output_dim,
        seq_len           = d.seq_len,
        horizon           = d.horizon,
        rnn_units         = m.rnn_units,
        num_rnn_layers    = m.num_rnn_layers,
        max_diffusion_step= m.max_diffusion_step,
        filter_type       = m.filter_type,
        cl_decay_steps    = m.cl_decay_steps,
    )
