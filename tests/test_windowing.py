"""Unit and property tests for Planck taper windowing and waveform integration."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from diffgw_core import planck_taper, torch_planck_taper, jax_planck_taper
from diffgw_core.constants import MSUN_SEC
from torchwave.approximants.taylorf2 import TaylorF2 as TorchTaylorF2
from jaxwave.approximants.taylorf2 import TaylorF2 as JaxTaylorF2

jax.config.update("jax_enable_x64", True)


class TestExactValues:
    """Verify exact boundary values and midpoint value across backends."""

    @pytest.mark.parametrize("f_start,f_end", [(20.0, 80.0), (100.0, 500.0), (1000.0, 1500.0)])
    def test_exact_values_scalar(self, f_start, f_end):
        f_mid = 0.5 * (f_start + f_end)

        # f <= f_start -> w = 1.0
        assert planck_taper(f_start - 10.0, f_start, f_end) == 1.0
        assert planck_taper(f_start, f_start, f_end) == 1.0

        # f = 0.5 * (f_start + f_end) -> w = 0.5
        assert np.isclose(planck_taper(f_mid, f_start, f_end), 0.5, atol=1e-15)

        # f >= f_end -> w = 0.0
        assert planck_taper(f_end, f_start, f_end) == 0.0
        assert planck_taper(f_end + 10.0, f_start, f_end) == 0.0

    @pytest.mark.parametrize("f_start,f_end", [(20.0, 80.0), (100.0, 500.0)])
    def test_exact_values_torch(self, f_start, f_end):
        f_mid = 0.5 * (f_start + f_end)
        f_t = torch.tensor(
            [f_start - 5.0, f_start, f_mid, f_end, f_end + 5.0],
            dtype=torch.float64,
        )
        w = torch_planck_taper(f_t, f_start, f_end)

        assert torch.allclose(
            w,
            torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0], dtype=torch.float64),
            atol=1e-15,
            rtol=0.0,
        )

    @pytest.mark.parametrize("f_start,f_end", [(20.0, 80.0), (100.0, 500.0)])
    def test_exact_values_jax(self, f_start, f_end):
        f_mid = 0.5 * (f_start + f_end)
        f_j = jnp.array(
            [f_start - 5.0, f_start, f_mid, f_end, f_end + 5.0],
            dtype=jnp.float64,
        )
        w = jax_planck_taper(f_j, f_start, f_end)

        np.testing.assert_allclose(
            np.asarray(w),
            np.array([1.0, 1.0, 0.5, 0.0, 0.0], dtype=np.float64),
            atol=1e-15,
            rtol=0.0,
        )


class TestTorchAutograd:
    """Verify PyTorch autodiff stability, boundary derivatives, and continuity."""

    def test_boundary_and_interior_derivatives(self):
        f_start, f_end = 20.0, 80.0
        f_mid = 0.5 * (f_start + f_end)

        f_vals = torch.tensor(
            [10.0, f_start, 35.0, f_mid, 65.0, f_end, 90.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        w = torch_planck_taper(f_vals, f_start, f_end)
        w.sum().backward()
        dw = f_vals.grad

        # Derivatives at and outside boundaries must be exactly 0.0
        assert dw[0].item() == 0.0
        assert dw[1].item() == 0.0
        assert dw[5].item() == 0.0
        assert dw[6].item() == 0.0

        # Derivatives inside (f_start, f_end) must be non-zero and negative (falling roll-off)
        assert dw[2].item() < 0.0
        assert dw[3].item() < 0.0
        assert dw[4].item() < 0.0

        # Analytical derivative at midpoint:
        # z = 0.5, dw/dz = -2, dz/df = 1 / (f_end - f_start)
        # dw/df = -2 / (f_end - f_start) = -2 / 60 = -1/30
        expected_mid_grad = -2.0 / (f_end - f_start)
        assert np.isclose(dw[3].item(), expected_mid_grad, atol=1e-14)

    def test_derivative_continuity_near_boundaries(self):
        f_start, f_end = 20.0, 80.0

        # Dense evaluation approaching boundaries from inside and outside
        epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        for eps in epsilons:
            f_near_start = torch.tensor([f_start - eps, f_start + eps], dtype=torch.float64, requires_grad=True)
            w_start = torch_planck_taper(f_near_start, f_start, f_end)
            w_start.sum().backward()
            g_left, g_right = f_near_start.grad[0].item(), f_near_start.grad[1].item()
            assert g_left == 0.0
            # Derivative is continuous: as eps -> 0, g_right approaches 0 smoothly
            assert abs(g_right) < 1.0

            f_near_end = torch.tensor([f_end - eps, f_end + eps], dtype=torch.float64, requires_grad=True)
            w_end = torch_planck_taper(f_near_end, f_start, f_end)
            w_end.sum().backward()
            g_in, g_out = f_near_end.grad[0].item(), f_near_end.grad[1].item()
            assert g_out == 0.0
            assert abs(g_in) < 1.0

    def test_parameter_gradients(self):
        # Gradients with respect to f_start and f_end
        f = torch.linspace(15.0, 85.0, 100, dtype=torch.float64, requires_grad=True)
        f_s = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
        f_e = torch.tensor(80.0, dtype=torch.float64, requires_grad=True)

        w = torch_planck_taper(f, f_s, f_e)
        loss = w.sum()
        grad_f, grad_s, grad_e = torch.autograd.grad(loss, (f, f_s, f_e))

        assert not torch.isnan(grad_f).any()
        assert not torch.isinf(grad_f).any()
        assert not torch.isnan(grad_s).any()
        assert not torch.isnan(grad_e).any()
        # Increasing f_s shifts window right, increasing w(f) for interior points
        assert grad_s.item() > 0.0
        # Increasing f_e expands window right, increasing w(f) for interior points
        assert grad_e.item() > 0.0


class TestJaxAutodiff:
    """Verify JAX JIT compilation, jax.grad, and vmap compatibility."""

    def test_jax_jit_equality(self):
        f_start, f_end = 30.0, 150.0
        f_arr = jnp.linspace(10.0, 200.0, 200, dtype=jnp.float64)

        jit_taper = jax.jit(jax_planck_taper)
        w_eager = jax_planck_taper(f_arr, f_start, f_end)
        w_jit = jit_taper(f_arr, f_start, f_end)

        np.testing.assert_allclose(np.asarray(w_jit), np.asarray(w_eager), atol=1e-15)

    def test_jax_grad(self):
        f_start, f_end = 20.0, 80.0
        f_mid = 0.5 * (f_start + f_end)

        f_vals = jnp.array([10.0, f_start, 35.0, f_mid, 65.0, f_end, 90.0], dtype=jnp.float64)

        grad_fn = jax.jit(jax.grad(lambda f: jnp.sum(jax_planck_taper(f, f_start, f_end))))
        dw = grad_fn(f_vals)

        assert dw[0] == 0.0
        assert dw[1] == 0.0
        assert dw[2] < 0.0
        assert dw[3] < 0.0
        assert dw[4] < 0.0
        assert dw[5] == 0.0
        assert dw[6] == 0.0

        expected_mid_grad = -2.0 / (f_end - f_start)
        assert np.isclose(float(dw[3]), expected_mid_grad, atol=1e-14)

    def test_jax_vmap(self):
        f_start = jnp.array([20.0, 30.0, 40.0], dtype=jnp.float64)
        f_end = jnp.array([80.0, 90.0, 100.0], dtype=jnp.float64)
        f = jnp.linspace(10.0, 120.0, 50, dtype=jnp.float64)

        vmapped_fn = jax.vmap(lambda s, e: jax_planck_taper(f, s, e))
        res = vmapped_fn(f_start, f_end)
        assert res.shape == (3, 50)


class TestCrossBackendAgreement:
    """Verify agreement across scalar/NumPy, PyTorch, and JAX implementations."""

    @pytest.mark.parametrize("f_start,f_end", [(20.0, 80.0), (100.0, 500.0), (500.0, 1200.0)])
    def test_cross_backend_agreement(self, f_start, f_end):
        grid = np.linspace(f_start * 0.5, f_end * 1.5, 2000, dtype=np.float64)

        w_np = planck_taper(grid, f_start, f_end)
        w_torch = torch_planck_taper(torch.from_numpy(grid), f_start, f_end).cpu().numpy()
        w_jax = np.asarray(jax_planck_taper(jnp.asarray(grid), f_start, f_end))

        # Check maximum absolute difference is strictly below 1e-12
        max_diff_torch_np = np.max(np.abs(w_torch - w_np))
        max_diff_jax_np = np.max(np.abs(w_jax - w_np))
        max_diff_torch_jax = np.max(np.abs(w_torch - w_jax))

        assert max_diff_torch_np < 1e-12, f"Torch vs NumPy diff {max_diff_torch_np} exceeds 1e-12"
        assert max_diff_jax_np < 1e-12, f"JAX vs NumPy diff {max_diff_jax_np} exceeds 1e-12"
        assert max_diff_torch_jax < 1e-12, f"Torch vs JAX diff {max_diff_torch_jax} exceeds 1e-12"


class TestWaveformIntegration:
    """Verify TaylorF2 waveform evaluation with window='planck' in Torch and JAX."""

    def test_taylorf2_torch_planck_window(self):
        f = torch.linspace(20.0, 400.0, 500, dtype=torch.float64)
        m1, m2 = 30.0, 20.0
        total_m_sec = (m1 + m2) * MSUN_SEC
        f_isco = 1.0 / (6.0 ** 1.5 * np.pi * total_m_sec)

        # Standard sharp cutoff
        hp_sharp, hc_sharp = TorchTaylorF2()(f, m1, m2, smooth_cutoff=False)
        # Planck window cutoff
        hp_planck, hc_planck = TorchTaylorF2()(f, m1, m2, window="planck", roll_off_delta=0.1)

        amp_sharp = hp_sharp.abs()
        amp_planck = hp_planck.abs()

        # Above ISCO, both are strictly zero
        idx_above = f >= f_isco
        assert torch.any(idx_above)
        assert torch.all(amp_sharp[idx_above] == 0.0)
        assert torch.all(amp_planck[idx_above] == 0.0)

        # Below roll-off start, amp_planck matches amp_sharp
        f_start = 0.9 * f_isco
        idx_below = (f <= f_start) & (f > 20.0)
        assert torch.allclose(amp_planck[idx_below], amp_sharp[idx_below], rtol=1e-12)

        # Inside taper window, amp_planck is strictly less than amp_sharp
        idx_window = (f > f_start) & (f < f_isco)
        assert torch.count_nonzero(idx_window) > 2
        assert torch.all(amp_planck[idx_window] < amp_sharp[idx_window])
        # Monotonically non-increasing in window
        diffs = torch.diff(amp_planck[idx_window])
        assert torch.all(diffs < 0.0)

    def test_taylorf2_jax_planck_window(self):
        f = jnp.linspace(20.0, 400.0, 500, dtype=jnp.float64)
        m1, m2 = 30.0, 20.0
        total_m_sec = (m1 + m2) * MSUN_SEC
        f_isco = 1.0 / (6.0 ** 1.5 * np.pi * total_m_sec)

        hp_sharp, hc_sharp = JaxTaylorF2()(f, m1, m2, smooth_cutoff=False)
        hp_planck, hc_planck = JaxTaylorF2()(f, m1, m2, window="planck", roll_off_delta=0.1)

        amp_sharp = jnp.abs(hp_sharp)
        amp_planck = jnp.abs(hp_planck)

        idx_above = f >= f_isco
        assert jnp.any(idx_above)
        assert jnp.all(amp_sharp[idx_above] == 0.0)
        assert jnp.all(amp_planck[idx_above] == 0.0)

        f_start = 0.9 * f_isco
        idx_below = (f <= f_start) & (f > 20.0)
        np.testing.assert_allclose(
            np.asarray(amp_planck[idx_below]),
            np.asarray(amp_sharp[idx_below]),
            rtol=1e-12,
        )

        idx_window = (f > f_start) & (f < f_isco)
        assert jnp.count_nonzero(idx_window) > 2
        assert jnp.all(amp_planck[idx_window] < amp_sharp[idx_window])
        assert jnp.all(jnp.diff(amp_planck[idx_window]) < 0.0)

    @pytest.mark.parametrize("window", [None, "cosine", "planck"])
    @pytest.mark.parametrize("chi1,chi2", [(0.7, -0.2), (-0.7, 0.2)])
    def test_kerr_cutoff_backend_parity(self, window, chi1, chi2):
        """Exercise both polarizations across the spinning cutoff and taper."""
        f = np.linspace(20.0, 220.0, 2001)
        options = dict(chi1=chi1, chi2=chi2, inclination=0.6,
                       isco_type="kerr", window=window)
        ht = TorchTaylorF2()(torch.from_numpy(f), 30.0, 20.0, **options)
        hj = JaxTaylorF2()(jnp.asarray(f), 30.0, 20.0, **options)
        for t, j in zip(ht, hj):
            t, j = np.asarray(t), np.asarray(j)
            assert np.count_nonzero(t) > 0
            assert np.count_nonzero(t == 0.0) > 0
            assert np.linalg.norm(t - j) / np.linalg.norm(t) < 1e-12

    def test_kerr_planck_spin_derivative(self):
        """Check the cutoff spin derivative against a scalar finite difference."""
        m1, m2, chi1, chi2 = 30.0, 20.0, 0.7, -0.2

        def cutoff(spin1):
            # Equatorial Kerr ISCO; the binary adapter uses mass-weighted spin.
            a = (m1 * spin1 + m2 * chi2) / (m1 + m2)
            z1 = 1.0 + np.cbrt(1.0 - a*a) * (np.cbrt(1.0 + a) + np.cbrt(1.0 - a))
            z2 = np.sqrt(3.0*a*a + z1*z1)
            r = 3.0 + z2 - np.sign(a)*np.sqrt((3.0-z1)*(3.0+z1+2.0*z2))
            return 1.0 / (np.pi*(m1+m2)*MSUN_SEC*(r**1.5+a))

        f = np.array([0.95 * cutoff(chi1)])
        spin_t = torch.tensor(chi1, dtype=torch.float64, requires_grad=True)
        opts = dict(chi2=chi2, isco_type="kerr", window="planck")
        h_t = TorchTaylorF2()(torch.from_numpy(f), m1, m2, chi1=spin_t, **opts)[0]
        h0_t = TorchTaylorF2()(torch.from_numpy(f), m1, m2, chi1=spin_t,
                              f_isco_cutoff=False)[0]
        weight_t = (h_t.abs() / h0_t.abs()).sum()
        grad_t = torch.autograd.grad(weight_t, spin_t)[0].item()

        def weight_j(spin1):
            h = JaxTaylorF2()(jnp.asarray(f), m1, m2, chi1=spin1, **opts)[0]
            h0 = JaxTaylorF2()(jnp.asarray(f), m1, m2, chi1=spin1,
                             f_isco_cutoff=False)[0]
            return jnp.sum(jnp.abs(h) / jnp.abs(h0))

        grad_j = float(jax.jit(jax.grad(weight_j))(chi1))
        eps = 1e-6
        weights = [planck_taper(f, 0.9*cutoff(s), cutoff(s))[0]
                   for s in (chi1-eps, chi1+eps)]
        expected = (weights[1] - weights[0]) / (2.0*eps)
        assert expected > 0.0
        np.testing.assert_allclose([grad_t, grad_j], expected, rtol=1e-8, atol=1e-10)

    def test_kerr_planck_batch_and_jit(self):
        f = np.linspace(20.0, 220.0, 501)
        params = ([30.0, 40.0], [20.0, 20.0], [0.7, -0.7], [-0.2, 0.2])
        pt = [torch.tensor(p, dtype=torch.float64) for p in params]
        pj = [jnp.asarray(p) for p in params]
        ht = TorchTaylorF2()(torch.from_numpy(f), pt[0], pt[1],
                            chi1=pt[2], chi2=pt[3], isco_type="kerr", window="planck")
        evaluate_j = jax.jit(lambda m1, m2, s1, s2: JaxTaylorF2()(
            jnp.asarray(f), m1, m2, chi1=s1, chi2=s2, isco_type="kerr", window="planck"))
        hj = evaluate_j(*pj)
        for t, j in zip(ht, hj):
            assert t.shape == j.shape == (2, len(f))
            errors = np.linalg.norm(np.asarray(t)-np.asarray(j), axis=1) / np.linalg.norm(np.asarray(t), axis=1)
            assert np.all(errors < 1e-12)
