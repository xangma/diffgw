# SPDX-License-Identifier: MIT
"""XHM Eq. (43)--(44) non-mixing ringdown phase and C1 attachment gate."""

import mpmath as mp
import numpy as np

from diffgw_reference import imrphenomxhm_nonmix_ringdown_phase as scalar


# (join, ring, damping, a2, a_lambda, phase_at_join, slope_at_join, grid)
CASES = (
    (".08", ".095", ".015", ".28", ".13", "2.7", "26", (".08", ".09", ".095", ".13", ".20")),
    (".10", ".12", ".018", "-.12", ".08", "-1.4", "18.5", (".10", ".11", ".12", ".16", ".25")),
    (".063", ".075", ".012", ".4", "-.07", ".8", "16.2", (".063", ".07", ".075", ".105", ".15")),
)

PHASE_RTOL, PHASE_ATOL = 5e-13, 2e-11
GRAD_RTOL, GRAD_ATOL = 2e-11, 2e-9


def _coupled_phase(frequency, join, ring, damp, a2, a_lambda, phase_at_join, slope_at_join):
    shifts = scalar.attachment_shifts(
        join, phase_at_join, slope_at_join, ring, damp, a2, a_lambda,
    )
    return scalar.phase(frequency, ring, damp, a2, a_lambda, *shifts)


def _reference():
    with mp.workdps(70):
        shifts, phase, slope, gradients = [], [], [], []
        for case in CASES:
            join, ring, damp, a2, a_lambda, p, g = map(mp.mpf, case[:7])
            row_shifts = scalar.attachment_shifts(join, p, g, ring, damp, a2, a_lambda)
            shifts.append([float(x) for x in row_shifts])
            row_phase, row_slope = [], []
            row_grads = [mp.mpf(0) for _ in range(4)]
            for f in map(mp.mpf, case[7]):
                args = (f, join, ring, damp, a2, a_lambda, p, g)
                row_phase.append(float(_coupled_phase(*args)))
                row_slope.append(float(scalar.derivative(f, ring, damp, a2, a_lambda, row_shifts[0])))
                for slot, index in enumerate((4, 5, 6, 7)):
                    row_grads[slot] += mp.diff(
                        lambda x: _coupled_phase(*(args[:index] + (x,) + args[index + 1:])),
                        args[index],
                    )
            phase.append(row_phase)
            slope.append(row_slope)
            gradients.append([float(x) for x in row_grads])
        return tuple(np.asarray(x, dtype=np.float64) for x in (shifts, phase, slope, gradients))


def _inputs():
    frequency = np.asarray([[float(x) for x in case[7]] for case in CASES])
    params = [np.asarray([float(case[i]) for case in CASES]) for i in range(7)]
    return (frequency, *params)


def test_scalar_antiderivative_and_c1_join():
    with mp.workdps(70):
        for case in CASES:
            join, ring, damp, a2, a_lambda, p, g = map(mp.mpf, case[:7])
            time_shift, phase_shift = scalar.attachment_shifts(join, p, g, ring, damp, a2, a_lambda)
            assert abs(scalar.phase(join, ring, damp, a2, a_lambda, time_shift, phase_shift) - p) < mp.mpf("1e-65")
            assert abs(scalar.derivative(join, ring, damp, a2, a_lambda, time_shift) - g) < mp.mpf("1e-65")
            for f in map(mp.mpf, case[7]):
                def phase_f(x):
                    return scalar.phase(x, ring, damp, a2, a_lambda, time_shift, phase_shift)

                def slope_f(x):
                    return scalar.derivative(x, ring, damp, a2, a_lambda, time_shift)

                assert abs(mp.diff(phase_f, f) - slope_f(f)) < mp.mpf("1e-55")
                assert abs(mp.diff(slope_f, f) - scalar.second_derivative(f, ring, damp, a2, a_lambda)) < mp.mpf("1e-55")


def test_torch_jax_batched_phase_shifts_and_gradients():
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import imrphenom_xhm as jax_xhm
    from torchwave.approximants import imrphenom_xhm as torch_xhm

    jax.config.update("jax_enable_x64", True)
    expected_shifts, expected_phase, expected_slope, expected_grads = _reference()
    frequency, join, ring, damp, a2, a_lambda, p, g = _inputs()

    t_f = torch.tensor(frequency, dtype=torch.float64, requires_grad=True)
    t_join, t_ring, t_damp = (torch.tensor(x, dtype=torch.float64) for x in (join, ring, damp))
    t_a2, t_alam, t_p, t_g = (
        torch.tensor(x, dtype=torch.float64, requires_grad=True)
        for x in (a2, a_lambda, p, g)
    )
    t_shifts = torch_xhm._nonmix_ringdown_phase_shifts(t_join, t_p, t_g, t_ring, t_damp, t_a2, t_alam)
    t_phase = torch_xhm._nonmix_ringdown_phase(t_f, t_ring, t_damp, t_a2, t_alam, *t_shifts)
    t_df, *t_grads = torch.autograd.grad(t_phase.sum(), (t_f, t_a2, t_alam, t_p, t_g), retain_graph=True)
    t_join_f = torch.tensor(join, dtype=torch.float64, requires_grad=True)
    t_join_phase = torch_xhm._nonmix_ringdown_phase(t_join_f, t_ring, t_damp, t_a2, t_alam, *t_shifts)
    (t_join_slope,) = torch.autograd.grad(t_join_phase.sum(), t_join_f)

    j_args = tuple(jnp.asarray(x) for x in (frequency, join, ring, damp, a2, a_lambda, p, g))

    def j_evaluate(f, j, r, d, a2_value, alam_value, p_value, g_value):
        t, c = jax_xhm._nonmix_ringdown_phase_shifts(j, p_value, g_value, r, d, a2_value, alam_value)
        return jax_xhm._nonmix_ringdown_phase(f, r, d, a2_value, alam_value, t, c)

    j_shifts = jax.jit(jax_xhm._nonmix_ringdown_phase_shifts)(
        j_args[1], j_args[6], j_args[7], j_args[2], j_args[3], j_args[4], j_args[5],
    )
    j_phase = jax.jit(j_evaluate)(*j_args)
    j_df, *j_grads = jax.jit(jax.grad(lambda *args: j_evaluate(*args).sum(), argnums=(0, 4, 5, 6, 7)))(*j_args)
    j_join_slope = jax.jit(jax.grad(
        lambda f: jax_xhm._nonmix_ringdown_phase(f, j_args[2], j_args[3], j_args[4], j_args[5], *j_shifts).sum(),
    ))(j_args[1])

    for actual in (np.stack([x.detach().numpy() for x in t_shifts], axis=-1),
                   np.stack([np.asarray(x) for x in j_shifts], axis=-1)):
        np.testing.assert_allclose(actual, expected_shifts, rtol=PHASE_RTOL, atol=PHASE_ATOL)
    for actual in (t_phase.detach().numpy(), np.asarray(j_phase)):
        np.testing.assert_allclose(actual, expected_phase, rtol=PHASE_RTOL, atol=PHASE_ATOL)
    for actual in (t_df.detach().numpy(), np.asarray(j_df)):
        np.testing.assert_allclose(actual, expected_slope, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    for actual in (np.stack([x.detach().numpy() for x in t_grads], axis=-1),
                   np.stack([np.asarray(x) for x in j_grads], axis=-1)):
        np.testing.assert_allclose(actual, expected_grads, rtol=GRAD_RTOL, atol=GRAD_ATOL)
    for phase_at_join, slope_at_join in ((t_join_phase.detach().numpy(), t_join_slope.detach().numpy()),
                                         (np.asarray(jax.jit(j_evaluate)(j_args[1], *j_args[1:])), np.asarray(j_join_slope))):
        np.testing.assert_allclose(phase_at_join, p, rtol=0, atol=2e-11)
        np.testing.assert_allclose(slope_at_join, g, rtol=0, atol=2e-9)
