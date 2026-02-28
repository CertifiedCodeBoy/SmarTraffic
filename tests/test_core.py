"""
Unit & integration tests for SmarTraffic.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ── Utils ─────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_masked_mae_basic(self):
        from src.utils import masked_mae

        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.0, 0.0, 4.0])   # 0 is masked
        loss = masked_mae(pred, true)
        assert abs(loss.item() - 1.0) < 1e-5    # |1-1| + |3-4| / 2 = 0.5... wait
        # non-zero indices: 0 and 2 → |1-1|=0, |3-4|=1 → mean=0.5
        assert abs(loss.item() - 0.5) < 1e-5

    def test_masked_mape(self):
        from src.utils import masked_mape

        pred = torch.tensor([10.0, 20.0])
        true = torch.tensor([10.0, 25.0])
        mape = masked_mape(pred, true)
        # 0% for sensor 0, 20% for sensor 1 → mean 10%
        assert abs(mape.item() - 10.0) < 1e-4

    def test_compute_all_metrics_keys(self):
        from src.utils import compute_all_metrics

        pred = torch.randn(4, 12, 207, 1)
        true = torch.randn(4, 12, 207, 1)
        m = compute_all_metrics(pred, true)
        assert set(m.keys()) == {"MAE", "RMSE", "MAPE"}
        for v in m.values():
            assert isinstance(v, float)
            assert v >= 0.0


class TestScaler:
    def test_fit_transform_inverse(self):
        from src.utils import StandardScaler

        x = np.random.randn(1000, 207, 5).astype(np.float32) * 30 + 50
        scaler = StandardScaler().fit(x)
        xn = scaler.transform(x)
        assert abs(xn.mean()) < 0.1
        assert abs(xn.std() - 1.0) < 0.1

        # Round-trip
        xr = scaler.inverse_transform(xn)
        np.testing.assert_allclose(xr, x, atol=1e-4)

    def test_torch_inverse(self):
        from src.utils import StandardScaler

        x = np.random.randn(100, 10, 2).astype(np.float32)
        scaler = StandardScaler().fit(x)
        xn = torch.from_numpy(scaler.transform(x))
        xr = scaler.inverse_transform(xn)
        assert isinstance(xr, torch.Tensor)


# ── Graph ─────────────────────────────────────────────────────────────────────

class TestGraphUtils:
    def test_build_support_shape(self):
        from src.utils import build_support

        N   = 20
        adj = np.random.rand(N, N).astype(np.float32)
        np.fill_diagonal(adj, 0)
        supports = build_support(adj, "dual_random_walk", max_diffusion_step=2)
        # identity + (2+1)*2 = 1 + 6 = 7
        assert len(supports) == 7
        for s in supports:
            assert s.shape == (N, N)

    def test_diffusion_matrices_row_stochastic(self):
        from src.utils import compute_diffusion_matrices

        N   = 15
        adj = np.abs(np.random.randn(N, N)).astype(np.float32)
        np.fill_diagonal(adj, 0)
        mats = compute_diffusion_matrices(adj, "dual_random_walk")
        for m in mats:
            row_sums = m.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones(N), atol=1e-5)


# ── Model ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_adj():
    N   = 15
    adj = np.abs(np.random.randn(N, N)).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


@pytest.fixture(scope="module")
def small_model(small_adj):
    from src.model import DCRNNModel

    return DCRNNModel(
        adj                = small_adj,
        input_dim          = 2,
        output_dim         = 1,
        seq_len            = 6,
        horizon            = 3,
        rnn_units          = 16,
        num_rnn_layers     = 1,
        max_diffusion_step = 1,
        filter_type        = "dual_random_walk",
    )


class TestModel:
    def test_forward_shape(self, small_model, small_adj):
        B, T, N, F = 4, 6, small_adj.shape[0], 2
        H = 3
        x = torch.randn(B, T, N, F)
        out = small_model(x)
        assert out.shape == (B, H, N, 1), f"Got {out.shape}"

    def test_train_vs_eval_shapes_match(self, small_model, small_adj):
        N = small_adj.shape[0]
        x = torch.randn(2, 6, N, 2)

        small_model.train()
        out_train = small_model(x)

        small_model.eval()
        with torch.no_grad():
            out_eval = small_model(x)

        assert out_train.shape == out_eval.shape

    def test_count_parameters(self, small_model):
        params = small_model.count_parameters()
        assert params > 0
        assert isinstance(params, int)

    def test_predict_no_grad(self, small_model, small_adj):
        N = small_adj.shape[0]
        x = torch.randn(1, 6, N, 2)
        out = small_model.predict(x)
        assert out.shape == (1, 3, N, 1)


class TestDiffusionConvolution:
    def test_output_shape(self):
        from src.model import DiffusionConvolution

        N, C_in, C_out = 10, 4, 8
        conv = DiffusionConvolution(C_in, C_out, num_supports=3)
        x    = torch.randn(2, N, C_in)
        supp = [torch.eye(N) for _ in range(3)]
        out  = conv(x, supp)
        assert out.shape == (2, N, C_out)


class TestDCGRUCell:
    def test_hidden_shape(self):
        from src.model import DCGRUCell
        from src.utils import build_support

        N   = 8
        adj = np.abs(np.random.randn(N, N)).astype(np.float32)
        np.fill_diagonal(adj, 0)
        supports_np = build_support(adj, "dual_random_walk", max_diffusion_step=1)
        supports    = [torch.from_numpy(s) for s in supports_np]

        cell = DCGRUCell(input_dim=3, units=16, max_diffusion_step=1, num_nodes=N)
        x    = torch.randn(2, N, 3)
        h    = torch.zeros(2, N, 16)
        h_new = cell(x, h, supports)
        assert h_new.shape == (2, N, 16)
