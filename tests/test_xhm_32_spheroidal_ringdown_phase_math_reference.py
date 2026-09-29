# SPDX-License-Identifier: MIT
"""XHM Eqs. (46)-(47) against an independent scalar primitive and derivatives."""

import mpmath as mp
import numpy as np

from diffgw_reference import imrphenomxhm_transition_mixing as scalar


PARAMS = (
    (".091", ".014", ".20", ".0001", ".30", "8", ".17"),
    (".115", ".018", ".13", "-.00008", "-.24", "4", "-.31"),
    (".078", ".012", "-.11", ".00004", ".41", "-3", ".09"),
)
FREQUENCIES = (
    (".070", ".084", ".091", ".112"),
    (".086", ".106", ".115", ".142"),
    (".061", ".073", ".078", ".101"),
)
VALUE_RTOL, VALUE_ATOL = 2e-13, 5e-12
JAC_RTOL, JAC_ATOL = 2e-12, 5e-10


def _scalar_output(frequency, params, output):
    ring, damp, a2, a4, a_lambda, time_shift, phase_shift = params
    coefficients = (a2, a4, a_lambda, time_shift)
    if output == 0:
        return scalar.ringdown_spheroidal_phase(
            frequency, ring, damp, coefficients, offset=phase_shift,
        )
    return scalar.ringdown_spheroidal_phase_derivative(frequency, ring, damp, coefficients)


def _scalar_expected():
    nrow, nfreq, noutput, nparam = len(PARAMS), len(FREQUENCIES[0]), 2, len(PARAMS[0])
    values = np.zeros((nrow, nfreq, noutput), dtype=np.float64)
    frequency_jac = np.zeros((nrow, nfreq, noutput, nrow, nfreq), dtype=np.float64)
    parameter_jac = np.zeros((nrow, nfreq, noutput, nrow, nparam), dtype=np.float64)
    with mp.workdps(70):
        for row, (raw_params, frequencies) in enumerate(zip(PARAMS, FREQUENCIES)):
            params = tuple(map(mp.mpf, raw_params))
            for index, raw_frequency in enumerate(frequencies):
                frequency = mp.mpf(raw_frequency)
                for output in range(noutput):
                    values[row, index, output] = float(_scalar_output(frequency, params, output))
                    frequency_jac[row, index, output, row, index] = float(mp.diff(
                        lambda f: _scalar_output(f, params, output), frequency,
                    ))
                    for param_index in range(nparam):
                        parameter_jac[row, index, output, row, param_index] = float(mp.diff(
                            lambda value: _scalar_output(
                                frequency,
                                params[:param_index] + (value,) + params[param_index + 1:],
                                output,
                            ),
                            params[param_index],
                        ))
                # The Eq. (47) primitive must differentiate to Eq. (46).
                np.testing.assert_allclose(
                    frequency_jac[row, index, 0, row, index], values[row, index, 1],
                    rtol=VALUE_RTOL, atol=VALUE_ATOL,
                )
    return values, frequency_jac, parameter_jac


def test_scalar_spheroidal_phase_shape_and_offset():
    values, frequency_jac, parameter_jac = _scalar_expected()
    assert np.isfinite(values).all()
    assert np.isfinite(frequency_jac).all()
    assert np.isfinite(parameter_jac).all()
    np.testing.assert_array_equal(parameter_jac[:, :, 0, :, -1].diagonal(axis1=0, axis2=2), 1.0)
    np.testing.assert_array_equal(parameter_jac[:, :, 1, :, -1], 0.0)


def test_torch_jax_spheroidal_phase_shape_values_and_full_jacobians():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import imrphenom_xhm as jax_xhm
    from torchwave.approximants import imrphenom_xhm as torch_xhm

    scalar_values, scalar_frequency_jac, scalar_parameter_jac = _scalar_expected()
    frequency = np.asarray(FREQUENCIES, dtype=np.float64)
    params = np.asarray(PARAMS, dtype=np.float64)

    def torch_evaluate(f, p):
        args = tuple(p[:, i] for i in range(6))
        phase = torch_xhm._spheroidal_32_ringdown_phase(f, *args, p[:, 6])
        derivative = torch_xhm._spheroidal_32_ringdown_phase_derivative(f, *args)
        return torch.stack((phase, derivative), dim=-1)

    def jax_evaluate(f, p):
        args = tuple(p[:, i] for i in range(6))
        phase = jax_xhm._spheroidal_32_ringdown_phase(f, *args, p[:, 6])
        derivative = jax_xhm._spheroidal_32_ringdown_phase_derivative(f, *args)
        return jnp.stack((phase, derivative), axis=-1)

    t_frequency = torch.tensor(frequency, dtype=torch.float64, requires_grad=True)
    t_params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    t_values = torch_evaluate(t_frequency, t_params).detach().numpy()
    t_frequency_jac, t_parameter_jac = (
        item.detach().numpy()
        for item in torch.autograd.functional.jacobian(torch_evaluate, (t_frequency, t_params))
    )
    j_frequency, j_params = jnp.asarray(frequency), jnp.asarray(params)
    j_values = np.asarray(jax.jit(jax_evaluate)(j_frequency, j_params))
    j_frequency_jac, j_parameter_jac = (
        np.asarray(item) for item in jax.jit(jax.jacrev(jax_evaluate, argnums=(0, 1)))(j_frequency, j_params)
    )

    for actual in (t_values, j_values):
        np.testing.assert_allclose(actual, scalar_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    for actual in (t_frequency_jac, j_frequency_jac):
        np.testing.assert_allclose(actual, scalar_frequency_jac, rtol=JAC_RTOL, atol=JAC_ATOL)
    for actual in (t_parameter_jac, j_parameter_jac):
        np.testing.assert_allclose(actual, scalar_parameter_jac, rtol=JAC_RTOL, atol=JAC_ATOL)
