# SPDX-License-Identifier: MIT
"""Published Eq. (42) XHM ringdown-amplitude component gate."""

import mpmath as mp
import numpy as np

from diffgw_reference import imrphenomxhm_ringdown_amplitude as scalar


CASES = (
    (".09", ".016", "1.3", ".015", ".4", (".06", ".08", ".09", ".11", ".15")),
    (".08", ".012", ".91", "-.023", ".65", (".05", ".07", ".08", ".10", ".14")),
)

VALUE_RTOL, VALUE_ATOL = 5e-13, 2e-12
GRAD_RTOL, GRAD_ATOL = 2e-11, 2e-10


def _reference():
    with mp.workdps(70):
        values, df, da, dl = [], [], [], []
        for ring, damp, sigma, a_lambda, lam, frequencies in CASES:
            value_row, df_row, da_row, dl_row = [], [], [], []
            for frequency in frequencies:
                args = (frequency, ring, damp, sigma, a_lambda, lam)
                value_row.append(float(scalar.amplitude_ratio(*args)))
                deriv = scalar.derivatives(*args)
                df_row.append(float(deriv[0]))
                da_row.append(float(deriv[1]))
                dl_row.append(float(deriv[2]))
            values.append(value_row)
            df.append(df_row)
            da.append(sum(da_row))
            dl.append(sum(dl_row))
        return tuple(np.asarray(x, dtype=np.float64) for x in (values, df, da, dl))


def _inputs():
    frequency = np.asarray([[float(x) for x in case[5]] for case in CASES])
    params = tuple(np.asarray([float(case[i]) for case in CASES]) for i in range(5))
    return (frequency, *params)


def test_independent_scalar_derivatives():
    with mp.workdps(70):
        for ring, damp, sigma, a_lambda, lam, frequencies in CASES:
            for frequency in frequencies:
                args = tuple(map(mp.mpf, (frequency, ring, damp, sigma, a_lambda, lam)))
                analytical = scalar.derivatives(*args)
                for index, derivative in ((0, 0), (4, 1), (5, 2)):
                    numerical = mp.diff(
                        lambda x: scalar.amplitude_ratio(*(args[:index] + (x,) + args[index + 1:])),
                        args[index],
                    )
                    assert abs(numerical - analytical[derivative]) < mp.mpf("1e-55")


def test_native_batched_values_and_gradients():
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import imrphenom_xhm as jax_xhm
    from torchwave.approximants import imrphenom_xhm as torch_xhm

    jax.config.update("jax_enable_x64", True)
    expected_value, expected_df, expected_da, expected_dl = _reference()
    frequency, ring, damp, sigma, a_lambda, lam = _inputs()

    t_f = torch.tensor(frequency, dtype=torch.float64, requires_grad=True)
    t_args = [torch.tensor(x, dtype=torch.float64) for x in (ring, damp, sigma)]
    t_a = torch.tensor(a_lambda, dtype=torch.float64, requires_grad=True)
    t_lam = torch.tensor(lam, dtype=torch.float64, requires_grad=True)
    t_value = torch_xhm._ringdown_amplitude_ratio(t_f, *t_args, t_a, t_lam)
    t_df, t_da, t_dl = torch.autograd.grad(t_value.sum(), (t_f, t_a, t_lam))

    j_args = tuple(jnp.asarray(x) for x in (frequency, ring, damp, sigma, a_lambda, lam))
    j_value = jax.jit(jax_xhm._ringdown_amplitude_ratio)(*j_args)
    j_df, j_da, j_dl = jax.jit(jax.grad(
        lambda *args: jax_xhm._ringdown_amplitude_ratio(*args).sum(), argnums=(0, 4, 5),
    ))(*j_args)

    for actual in (t_value.detach().numpy(), np.asarray(j_value)):
        np.testing.assert_allclose(actual, expected_value, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    for actual, expected in (
        (t_df.detach().numpy(), expected_df), (np.asarray(j_df), expected_df),
        (t_da.detach().numpy(), expected_da), (np.asarray(j_da), expected_da),
        (t_dl.detach().numpy(), expected_dl), (np.asarray(j_dl), expected_dl),
    ):
        np.testing.assert_allclose(actual, expected, rtol=GRAD_RTOL, atol=GRAD_ATOL)
