# SPDX-License-Identifier: MIT
"""XHM Eq. (37) scalar reference against native Torch and JAX algebra."""

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference import imrphenomxhm_intermediate_amplitude as ref


# f_start, f_end, v_start, dv_start/df, v_1, v_2, v_end, dv_end/df, A0
CASES = (
    ("0.03", "0.09", "0.90", "-1.2", "0.86", "0.81", "0.75", "-1.0", "0.7"),
    ("0.04", "0.12", "0.53", "0.4", "0.56", "0.59", "0.62", "0.3", "1.1"),
)


def _reference(case):
    lo, hi, start, ds, i1, i2, end, de, a0 = map(mp.mpf, case)
    coeffs = ref.coefficients(lo, hi, start, ds, (i1, i2), end, de)
    return lo, hi, a0, coeffs


def _torch_coefficients(case):
    import torch
    from torchwave.approximants import xhm_intermediate_amplitude_clean as tm

    lo, hi, start, ds, i1, i2, end, de, _ = map(float, case)
    def t(x):
        return torch.tensor(x, dtype=torch.float64)
    return tm.solve_intermediate_normalized_amplitude(
        t(start), t(ds), t((i1, i2)), t(end), t(de), t(lo), t(hi)
    )


def _jax_coefficients(case):
    import jax
    import jax.numpy as jnp
    from jaxwave.approximants import xhm_intermediate_amplitude_clean as jm

    jax.config.update("jax_enable_x64", True)
    lo, hi, start, ds, i1, i2, end, de, _ = map(float, case)
    return jax.jit(jm.solve_intermediate_normalized_amplitude)(
        start, ds, jnp.asarray((i1, i2)), end, de, lo, hi
    )


@pytest.mark.parametrize("case", CASES)
def test_scalar_reference_satisfies_paper_collocation_and_c1(case):
    with mp.workdps(70):
        lo, hi, a0, coeffs = _reference(case)
        nodes = (lo, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi)
        normalized = tuple(map(mp.mpf, (case[2], case[4], case[5], case[6])))
        for f, v in zip(nodes, normalized):
            assert mp.almosteq(ref.amplitude(f, a0, coeffs) / a0, v, abs_eps=mp.mpf("1e-55"))
        for f, target in ((lo, mp.mpf(case[3])), (hi, mp.mpf(case[7]))):
            assert mp.almosteq(ref.amplitude_derivative_df(f, a0, coeffs) / a0,
                               target, abs_eps=mp.mpf("1e-55"))
            assert mp.almosteq(mp.diff(lambda x: ref.amplitude(x, a0, coeffs), f),
                               ref.amplitude_derivative_df(f, a0, coeffs),
                               abs_eps=mp.mpf("1e-55"))


@pytest.mark.parametrize("case", CASES)
def test_native_values_frequency_derivatives_and_target_sensitivity(case):
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import xhm_intermediate_amplitude_clean as jm
    from torchwave.approximants import xhm_intermediate_amplitude_clean as tm

    jax.config.update("jax_enable_x64", True)
    with mp.workdps(70):
        lo, hi, a0, scalar_coeffs = _reference(case)
        frequencies = np.linspace(float(lo), float(hi), 11)
        expected = np.array([float(ref.amplitude(mp.mpf(str(f)), a0, scalar_coeffs))
                             for f in frequencies])
        expected_df = np.array([float(ref.amplitude_derivative_df(mp.mpf(str(f)), a0, scalar_coeffs))
                                for f in frequencies])
        midpoint = (lo + hi) / 2
        step = mp.mpf("1e-20")

        def at_target(v1):
            changed = list(case)
            changed[4] = v1
            return ref.amplitude(midpoint, a0, _reference(changed)[3])

        expected_target = float((at_target(mp.mpf(case[4]) + step) -
                                 at_target(mp.mpf(case[4]) - step)) / (2 * step))

    tc = _torch_coefficients(case)
    jc = _jax_coefficients(case)
    tf = torch.tensor(frequencies, dtype=torch.float64, requires_grad=True)
    actual_t = tm.evaluate_intermediate_amplitude(tf, float(lo), float(hi), float(a0), tc)
    df_t = torch.autograd.grad(actual_t.sum(), tf)[0]
    actual_j = jax.jit(jm.evaluate_intermediate_amplitude)(
        jnp.asarray(frequencies), float(lo), float(hi), float(a0), jc)
    df_j = jax.jit(jax.vmap(jax.grad(
        lambda f: jm.evaluate_intermediate_amplitude(f, float(lo), float(hi), float(a0), jc)
    )))(jnp.asarray(frequencies))
    for got in (np.asarray(actual_t.detach()), np.asarray(actual_j)):
        np.testing.assert_allclose(got, expected, rtol=2e-11, atol=2e-11)
    for got in (np.asarray(df_t), np.asarray(df_j)):
        np.testing.assert_allclose(got, expected_df, rtol=1e-8, atol=2e-9)

    def torch_target(value):
        data = list(map(float, case))
        c = tm.solve_intermediate_normalized_amplitude(
            data[2], data[3], torch.stack((value, torch.tensor(data[5], dtype=torch.float64))),
            data[6], data[7], data[0], data[1])
        return tm.evaluate_intermediate_amplitude(float(midpoint), data[0], data[1], data[8], c)

    target = torch.tensor(float(case[4]), dtype=torch.float64, requires_grad=True)
    grad_t = torch.autograd.grad(torch_target(target), target)[0]
    data = list(map(float, case))

    def jax_target(value):
        c = jm.solve_intermediate_normalized_amplitude(
            data[2], data[3], jnp.stack((value, jnp.asarray(data[5]))),
            data[6], data[7], data[0], data[1])
        return jm.evaluate_intermediate_amplitude(float(midpoint), data[0], data[1], data[8], c)

    grad_j = jax.jit(jax.grad(jax_target))(jnp.asarray(data[4]))
    np.testing.assert_allclose((float(grad_t), float(grad_j)), expected_target,
                               rtol=2e-7, atol=2e-8)


def test_native_batched_solve_and_join_grid():
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import xhm_intermediate_amplitude_clean as jm
    from torchwave.approximants import xhm_intermediate_amplitude_clean as tm

    jax.config.update("jax_enable_x64", True)
    data = np.array(CASES, dtype=np.float64)
    grid = data[:, 0, None] + (data[:, 1] - data[:, 0])[:, None] * np.linspace(0, 1, 9)
    with mp.workdps(70):
        expected = np.array([[float(ref.amplitude(mp.mpf(str(f)), _reference(case)[2],
                                                    _reference(case)[3])) for f in row]
                             for case, row in zip(CASES, grid)])
    td = torch.tensor(data)
    tc = tm.solve_intermediate_normalized_amplitude(
        td[:, 2], td[:, 3], td[:, 4:6], td[:, 6], td[:, 7], td[:, 0], td[:, 1])
    actual_t = tm.evaluate_intermediate_amplitude(torch.tensor(grid), td[:, 0, None],
        td[:, 1, None], td[:, 8, None], tc[:, None, :])
    jd = jnp.asarray(data)
    jc = jax.jit(jm.solve_intermediate_normalized_amplitude)(
        jd[:, 2], jd[:, 3], jd[:, 4:6], jd[:, 6], jd[:, 7], jd[:, 0], jd[:, 1])
    actual_j = jax.jit(jm.evaluate_intermediate_amplitude)(
        jnp.asarray(grid), jd[:, 0, None], jd[:, 1, None], jd[:, 8, None], jc[:, None, :])
    assert actual_t.shape == actual_j.shape == (2, 9)
    np.testing.assert_allclose(np.asarray(actual_t), expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(np.asarray(actual_j), expected, rtol=2e-11, atol=2e-11)
