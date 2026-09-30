"""Favata/Talbot dominant-22 source, scalar, and frontend qualification gates."""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from diffgw_reference import memory_dominant_22 as ref
from jaxwave.approximants import memory as jm
from torchwave.approximants import memory as tm

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


def test_nonuniform_nonlinear_direct_class_against_independent_sampled_scalar():
    """Exercise the numerical route beyond the exact linear-mode source gate."""
    angle = 0.73
    quadratic_real = 1.1e-20

    def scalar_at(coefficient):
        tt = tuple(mp.mpf(str(t)) for t in TIMES)
        samples = tuple((mp.mpc(str(SLOPE_REAL), str(SLOPE_IMAG)) * t
                         + mp.mpc(coefficient, "-0.5e-20") * t * t) for t in tt)
        return ref.sampled_h20(tt, samples, DISTANCE_M)

    with mp.workdps(70):
        expected_h20 = np.array([float(value) for value in scalar_at(mp.mpf(str(quadratic_real)))])
        expected_hp = np.array([float(ref.plus_from_h20(value, angle)) for value in expected_h20])
        expected_gradient = float(mp.diff(lambda re: scalar_at(re)[-1], mp.mpf(str(quadratic_real))))

    tt = torch.tensor(TIMES, dtype=torch.float64)
    real_t = torch.tensor(quadratic_real, dtype=torch.float64, requires_grad=True)
    mode_t = (SLOPE_REAL + 1j * SLOPE_IMAG) * tt + (real_t - 0.5e-20j) * tt.square()
    hp_t, hc_t, h20_t = tm.MemoryWaveform().evaluate_22(tt, mode_t, DISTANCE_M, angle)
    gradient_t = torch.autograd.grad(h20_t[-1], real_t)[0]

    tj = jnp.asarray(TIMES)
    def evaluate_j(re):
        mode = (SLOPE_REAL + 1j * SLOPE_IMAG) * tj + (re - 0.5e-20j) * tj**2
        return jm.MemoryWaveform().evaluate_22(tj, mode, DISTANCE_M, angle)
    hp_j, hc_j, h20_j = jax.jit(evaluate_j)(quadratic_real)
    gradient_j = jax.grad(lambda re: evaluate_j(re)[2][-1])(quadratic_real)

    for actual in (h20_t.detach().numpy(), np.asarray(h20_j)):
        np.testing.assert_allclose(actual, expected_h20, rtol=RTOL, atol=ATOL)
    for actual in (hp_t.detach().numpy(), np.asarray(hp_j)):
        np.testing.assert_allclose(actual, expected_hp, rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(hc_t.detach().numpy(), np.zeros_like(TIMES))
    np.testing.assert_array_equal(np.asarray(hc_j), np.zeros_like(TIMES))
    np.testing.assert_allclose([gradient_t.item(), float(gradient_j)], expected_gradient, rtol=RTOL, atol=0)


def test_sampled_scalar_rejects_invalid_domain():
    with pytest.raises(ValueError, match="increasing"):
        ref.sampled_h20((0, 0), (0j, 1j), DISTANCE_M)
    with pytest.raises(ValueError, match="positive"):
        ref.sampled_h20((0, 1), (0j, 1j), 0)


def test_two_sample_minimum_matches_exact_linear_reference():
    times = (0.0, 0.5)
    mode = (0j, complex(SLOPE_REAL, SLOPE_IMAG) * 0.5)
    expected = np.array([float(ref.h20_linear_mode(t, 0, complex(SLOPE_REAL, SLOPE_IMAG), DISTANCE_M))
                         for t in times])
    torch_h20 = tm.MemoryWaveform().evaluate_22(
        torch.tensor(times, dtype=torch.float64),
        torch.tensor(mode, dtype=torch.complex128), DISTANCE_M, 0.4)[2]
    jax_h20 = jax.jit(jm.MemoryWaveform().evaluate_22)(
        jnp.asarray(times), jnp.asarray(mode), DISTANCE_M, 0.4)[2]
    for actual in (torch_h20.detach().numpy(), np.asarray(jax_h20)):
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("batch_size", (2, len(TIMES)))
def test_batched_distances_match_scalar_calls_and_distance_gradients(batch_size):
    """Catch both incompatible B != N and silently column-scaled B == N."""
    scales = np.arange(1, batch_size + 1, dtype=np.float64)
    modes = scales[:, None] * (SLOPE_REAL + 1j * SLOPE_IMAG) * TIMES
    distances = DISTANCE_M * (1 + scales / 3)
    tt = torch.tensor(TIMES)
    mode_t = torch.tensor(modes)
    distance_t = torch.tensor(distances, requires_grad=True)
    h20_t = tm.MemoryWaveform().evaluate_22(tt, mode_t, distance_t, 0.4)[2]
    scalar_t = torch.stack([
        tm.compute_memory_h20_22(tt, mode_t[i], float(distances[i]))
        for i in range(batch_size)
    ])
    gradient_t = torch.autograd.grad(h20_t[:, -1].sum(), distance_t)[0]

    tj, mode_j, distance_j = map(jnp.asarray, (TIMES, modes, distances))
    h20_j = jax.jit(jm.MemoryWaveform().evaluate_22)(tj, mode_j, distance_j, 0.4)[2]
    scalar_j = jnp.stack([
        jm.compute_memory_h20_22(tj, mode_j[i], float(distances[i]))
        for i in range(batch_size)
    ])
    gradient_j = jax.jit(jax.grad(
        lambda distance: jm.compute_memory_h20_22(tj, mode_j, distance)[:, -1].sum()
    ))(distance_j)

    with mp.workdps(70):
        expected = np.array([
            [float(ref.h20_linear_mode(t, 0, scale * complex(SLOPE_REAL, SLOPE_IMAG), distance))
             for t in TIMES]
            for scale, distance in zip(scales, distances)
        ])
    for actual in (h20_t.detach().numpy(), np.asarray(h20_j), scalar_t.numpy(), np.asarray(scalar_j)):
        assert actual.shape == modes.shape
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    expected_gradient = expected[:, -1] / distances
    for actual in (gradient_t.numpy(), np.asarray(gradient_j)):
        np.testing.assert_allclose(actual, expected_gradient, rtol=RTOL, atol=0)


@pytest.mark.parametrize("shared_distance", (False, True))
def test_distance_broadcasts_only_over_leading_batch_dimensions(shared_distance):
    scales = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    modes = scales[..., None] * (SLOPE_REAL + 1j * SLOPE_IMAG) * TIMES
    distances = DISTANCE_M if shared_distance else DISTANCE_M * np.array([1., 2., 3.])
    distance_grid = np.broadcast_to(distances, scales.shape)
    with mp.workdps(70):
        expected = np.array([
            [float(ref.h20_linear_mode(t, 0, scale * complex(SLOPE_REAL, SLOPE_IMAG), distance))
             for t in TIMES]
            for scale, distance in zip(scales.flat, distance_grid.flat)
        ]).reshape(modes.shape)
    actual_t = tm.compute_memory_h20_22(
        torch.tensor(TIMES), torch.tensor(modes), torch.tensor(distances, dtype=torch.float64))
    actual_j = jax.jit(jm.compute_memory_h20_22)(
        jnp.asarray(TIMES), jnp.asarray(modes), jnp.asarray(distances))
    for actual in (actual_t.numpy(), np.asarray(actual_j)):
        assert actual.shape == modes.shape
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("batch_shape", ((), (2,)))
def test_per_time_distance_arrays_are_rejected(batch_shape):
    modes = np.broadcast_to((SLOPE_REAL + 1j * SLOPE_IMAG) * TIMES, batch_shape + TIMES.shape)
    distances = np.full(modes.shape, DISTANCE_M)
    with pytest.raises(ValueError, match="leading batch shape"):
        tm.compute_memory_h20_22(torch.tensor(TIMES), torch.tensor(modes), torch.tensor(distances))
    with pytest.raises(ValueError, match="leading batch shape"):
        jax.jit(jm.compute_memory_h20_22)(jnp.asarray(TIMES), jnp.asarray(modes), jnp.asarray(distances))
