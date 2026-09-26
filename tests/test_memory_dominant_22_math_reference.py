"""Favata/Talbot dominant-22 source, scalar, and frontend qualification gates."""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import torch

from diffgw_reference import memory_dominant_22 as ref
from jaxwave.approximants import memory as jm
from torchwave.approximants import memory as tm

jax.config.update("jax_enable_x64", True)

TIMES = np.array([0.0, 0.1, 0.4, 1.0], dtype=np.float64)
DISTANCE_M = 3.085677581491367e24  # 100 Mpc, explicitly in meters.
SLOPE_REAL = 2e-20
SLOPE_IMAG = 3e-20
INCLINATIONS = (0.0, 0.4, math.pi / 2, 2.1)
RTOL = 5e-13
ATOL = 1e-35


def test_favata_projection_and_full_pattern_identity():
    with mp.workdps(70):
        assert mp.almosteq(ref.h20_coefficient(), ref.h20_coefficient_by_projection(), rel_eps=mp.mpf("1e-65"))
        for angle in INCLINATIONS:
            h20 = ref.h20_from_flux_integral(mp.mpf("1e-40"), DISTANCE_M)
            projected = ref.plus_from_h20(h20, angle)
            direct = ref.plus_from_flux_integral(mp.mpf("1e-40"), DISTANCE_M, angle)
            assert mp.almosteq(projected, direct, rel_eps=mp.mpf("1e-65"), abs_eps=mp.mpf("1e-80"))
        h20 = ref.h20_from_flux_integral(mp.mpf("1e-40"), DISTANCE_M)
        h20_only = h20 * ref.spin_minus_two_y20(mp.pi / 2)
        assert not mp.almosteq(ref.plus_from_h20(h20, mp.pi / 2), h20_only)


@pytest.mark.parametrize("angle", INCLINATIONS)
def test_published_22_values_and_slopes_against_both_frontends(angle):
    with mp.workdps(70):
        expected_h20 = np.array([
            float(ref.h20_linear_mode(t, 0, complex(SLOPE_REAL, SLOPE_IMAG), DISTANCE_M))
            for t in TIMES
        ])
        expected_hp = np.array([float(ref.plus_from_h20(h, angle)) for h in expected_h20])
        expected_slope_grad = float(ref.dh20_d_real_slope(1, 0, SLOPE_REAL, SLOPE_IMAG, DISTANCE_M))
        expected_angle_grad = float(mp.diff(
            lambda inc: ref.plus_from_h20(ref.h20_linear_mode(1, 0, complex(SLOPE_REAL, SLOPE_IMAG), DISTANCE_M), inc),
            mp.mpf(str(angle)),
        ))

    tt = torch.tensor(TIMES, dtype=torch.float64)
    real_t = torch.tensor(SLOPE_REAL, dtype=torch.float64, requires_grad=True)
    angle_t = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
    h22_t = (real_t + 1j * SLOPE_IMAG) * tt
    hp_t, hc_t, h20_t = tm.MemoryWaveform().evaluate_22(tt, h22_t, DISTANCE_M, angle_t)
    slope_grad_t = torch.autograd.grad(h20_t[-1], real_t, retain_graph=True)[0]
    angle_grad_t = torch.autograd.grad(hp_t[-1], angle_t)[0]

    tj = jnp.asarray(TIMES)
    h22_j = (SLOPE_REAL + 1j * SLOPE_IMAG) * tj
    hp_j, hc_j, h20_j = jax.jit(jm.MemoryWaveform().evaluate_22)(tj, h22_j, DISTANCE_M, angle)
    slope_grad_j = jax.grad(lambda re: jm.compute_memory_h20_22(tj, (re + 1j * SLOPE_IMAG) * tj, DISTANCE_M)[-1])(SLOPE_REAL)
    angle_grad_j = jax.grad(lambda inc: jm.MemoryWaveform().evaluate_22(tj, h22_j, DISTANCE_M, inc)[0][-1])(angle)

    for actual in (h20_t.detach().numpy(), np.asarray(h20_j)):
        np.testing.assert_allclose(actual, expected_h20, rtol=RTOL, atol=ATOL)
    for actual in (hp_t.detach().numpy(), np.asarray(hp_j)):
        np.testing.assert_allclose(actual, expected_hp, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(hc_t.detach().numpy(), np.zeros_like(TIMES))
    np.testing.assert_array_equal(np.asarray(hc_j), np.zeros_like(TIMES))
    np.testing.assert_allclose([slope_grad_t.item(), float(slope_grad_j)], expected_slope_grad, rtol=RTOL, atol=0)
    np.testing.assert_allclose([angle_grad_t.item(), float(angle_grad_j)], expected_angle_grad, rtol=RTOL, atol=ATOL)
