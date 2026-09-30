# SPDX-License-Identifier: MIT
"""Paper Eq. (49)-(50) alignment, conditional on caller-supplied fit targets."""

import mpmath as mp
import numpy as np

from diffgw_reference import imrphenomxhm_transition_mixing as scalar


ROWS = (
    (".083", ".013", ".091", ".014", ".20", ".0001", ".30", "8", "11", "-.4", "2", ".15"),
    (".104", ".016", ".115", ".018", ".13", "-.00008", "-.24", "4", "-7", "1.2", "-1", "-.3"),
    (".072", ".010", ".078", ".012", "-.11", ".00004", ".41", "-3", "5", ".2", ".6", "-.1"),
)
VALUE_RTOL, VALUE_ATOL = 2e-13, 5e-12
JAC_RTOL, JAC_ATOL = 2e-12, 5e-10


def _expected():
    values = np.zeros((3, 2), dtype=np.float64)
    jacobian = np.zeros((3, 2, 3, 12), dtype=np.float64)
    with mp.workdps(70):
        for row_index, raw in enumerate(ROWS):
            row = tuple(map(mp.mpf, raw))
            result = scalar.ringdown_spheroidal_alignment(*row)
            values[row_index] = tuple(map(float, result))
            for output in range(2):
                for param in range(12):
                    jacobian[row_index, output, row_index, param] = float(mp.diff(
                        lambda value: scalar.ringdown_spheroidal_alignment(
                            *(row[:param] + (value,) + row[param + 1:]),
                        )[output],
                        row[param],
                    ))
            f0 = row[0] + row[1]
            f1 = row[0]
            coeffs = (*row[4:7], result[0])
            assert mp.almosteq(
                scalar.ringdown_spheroidal_phase_derivative(f0, row[2], row[3], coeffs) - row[8],
                row[10], abs_eps=mp.mpf("1e-60"),
            )
            assert mp.almosteq(
                scalar.ringdown_spheroidal_phase(f1, row[2], row[3], coeffs, offset=result[1]) - row[9],
                row[11], abs_eps=mp.mpf("1e-60"),
            )
    return values, jacobian


def test_scalar_alignment_identities_and_base_time_cancellation():
    values, jacobian = _expected()
    assert np.isfinite(values).all()
    assert np.isfinite(jacobian).all()
    np.testing.assert_array_equal(jacobian[:, :, :, 7], 0.0)


def test_torch_jax_production_alignment_values_and_full_jacobians():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import imrphenom_xhm as jax_xhm
    from torchwave.approximants import imrphenom_xhm as torch_xhm

    expected_values, expected_jacobian = _expected()
    inputs = np.asarray(ROWS, dtype=np.float64)

    def torch_evaluate(rows):
        offsets = torch_xhm._spheroidal_32_alignment(*(rows[:, i] for i in range(12)))
        return torch.stack(offsets, dim=-1)

    def jax_evaluate(rows):
        offsets = jax_xhm._spheroidal_32_alignment(*(rows[:, i] for i in range(12)))
        return jnp.stack(offsets, axis=-1)

    t_rows = torch.tensor(inputs, dtype=torch.float64, requires_grad=True)
    t_traced = torch.jit.trace(torch_evaluate, t_rows.detach(), check_trace=True)
    t_values = t_traced(t_rows).detach().numpy()
    t_jacobian = torch.autograd.functional.jacobian(t_traced, t_rows).detach().numpy()
    t_eager_values = torch_evaluate(t_rows).detach().numpy()
    t_eager_jacobian = torch.autograd.functional.jacobian(torch_evaluate, t_rows).detach().numpy()
    j_rows = jnp.asarray(inputs)
    j_values = np.asarray(jax.jit(jax_evaluate)(j_rows))
    j_jacobian = np.asarray(jax.jit(jax.jacrev(jax_evaluate))(j_rows))

    for actual in (t_values, t_eager_values, j_values):
        np.testing.assert_allclose(actual, expected_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    for actual in (t_jacobian, t_eager_jacobian, j_jacobian):
        np.testing.assert_allclose(actual, expected_jacobian, rtol=JAC_RTOL, atol=JAC_ATOL)
