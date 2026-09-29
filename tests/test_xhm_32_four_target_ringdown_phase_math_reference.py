# SPDX-License-Identifier: MIT
"""Published XHM Eq. (48) four-target solve on supplied target values."""

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference import imrphenomxhm_transition_mixing as scalar


ROWS = (
    (".080", ".095", ".012", "42", "39", "35", "31"),
    (".082", ".110", ".014", "-10", "-9", "-7", "-6"),
    (".070", ".087", ".010", "15", "16", "18", "20"),
)
VALUE_RTOL, VALUE_ATOL = 2e-11, 5e-10
JAC_RTOL, JAC_ATOL = 2e-9, 5e-7


def _coefficients(row):
    return scalar.ringdown_spheroidal_four_target_coefficients(*row[:3], row[3:])


def _scalar_expected():
    values = np.zeros((3, 4), dtype=np.float64)
    jacobian = np.zeros((3, 4, 3, 7), dtype=np.float64)
    conditions = []
    with mp.workdps(70):
        for row_index, raw in enumerate(ROWS):
            row = tuple(map(mp.mpf, raw))
            nodes, matrix = scalar.ringdown_spheroidal_four_target_system(*row[:3])
            coeff = _coefficients(row)
            values[row_index] = tuple(map(float, coeff))
            for i, (node, target) in enumerate(zip(nodes, row[3:])):
                actual = scalar.ringdown_spheroidal_phase_derivative(node, row[1], row[2], coeff)
                assert mp.almosteq(actual, target, abs_eps=mp.mpf("1e-60"))
            for output in range(4):
                for param in range(7):
                    jacobian[row_index, output, row_index, param] = float(mp.diff(
                        lambda value: _coefficients(
                            row[:param] + (value,) + row[param + 1:],
                        )[output],
                        row[param],
                    ))
            scaled_matrix = np.asarray(matrix.tolist(), dtype=np.float64)
            assert np.linalg.matrix_rank(scaled_matrix) == 4
            conditions.append(np.linalg.cond(scaled_matrix))
    return values, jacobian, np.asarray(conditions)


def test_scalar_four_target_residual_rank_conditioning_and_domain():
    values, jacobian, conditions = _scalar_expected()
    assert np.isfinite(values).all()
    assert np.isfinite(jacobian).all()
    np.testing.assert_array_less(conditions, 1000.0)
    np.testing.assert_array_less(100.0, conditions)
    with pytest.raises(ValueError, match="distinct"):
        scalar.ringdown_spheroidal_four_target_coefficients(
            ".077", ".095", ".012", ("1", "2", "3", "4"),
        )
    with pytest.raises(ValueError, match="positive"):
        scalar.ringdown_spheroidal_four_target_coefficients(
            ".080", ".014", ".012", ("1", "2", "3", "4"),
        )
    with mp.workdps(70):
        _, near = scalar.ringdown_spheroidal_four_target_system(".0770001", ".095", ".012")
    assert np.linalg.cond(np.asarray(near.tolist(), dtype=np.float64)) > 1e6


def test_standalone_torch_jax_four_target_values_and_full_jacobians():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import xhm_phase_clean as jax_phase
    from torchwave.approximants import xhm_phase_clean as torch_phase

    expected_values, expected_jacobian, _ = _scalar_expected()
    rows = np.asarray(ROWS, dtype=np.float64)

    def torch_evaluate(data):
        return torch_phase.xhm_32_spheroidal_phase_coefficients(
            data[:, 0], data[:, 1], data[:, 2], data[:, 3:],
        )

    def jax_evaluate(data):
        return jax_phase.xhm_32_spheroidal_phase_coefficients(
            data[:, 0], data[:, 1], data[:, 2], data[:, 3:],
        )

    t_rows = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
    t_values = torch_evaluate(t_rows).detach().numpy()
    t_jacobian = torch.autograd.functional.jacobian(torch_evaluate, t_rows).detach().numpy()
    j_rows = jnp.asarray(rows)
    j_values = np.asarray(jax.jit(jax_evaluate)(j_rows))
    j_jacobian = np.asarray(jax.jit(jax.jacrev(jax_evaluate))(j_rows))
    for actual in (t_values, j_values):
        np.testing.assert_allclose(actual, expected_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    for actual in (t_jacobian, j_jacobian):
        np.testing.assert_allclose(actual, expected_jacobian, rtol=JAC_RTOL, atol=JAC_ATOL)

    with pytest.raises(ValueError, match="distinct"):
        torch_phase.xhm_32_spheroidal_phase_coefficients(
            .077, .095, .012, [1., 2., 3., 4.],
        )
