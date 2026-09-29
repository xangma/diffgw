# SPDX-License-Identifier: MIT
"""XHM Eq. (36)/Table 2 default ringdown cuts against a scalar reference."""

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference import imrphenomxhm_ringdown_cuts as scalar


MODES = ("22", "21", "33", "32", "44")
OUTPUT_MODES = ("21", "33", "32", "44")
ROWS = (
    (".080", ".075", ".095", ".091", ".120", ".012", ".012", ".015", ".014", ".018"),
    (".087", ".079", ".100", ".095", ".130", ".010", ".010", ".013", ".012", ".016"),
    (".075", ".072", ".088", ".086", ".110", ".015", ".014", ".018", ".017", ".020"),
)
VALUE_ATOL = 2e-15
JAC_ATOL = 2e-14


def _expected():
    with mp.workdps(70):
        values = []
        for row in ROWS:
            ring = dict(zip(MODES, map(mp.mpf, row[:5])))
            damp = dict(zip(MODES, map(mp.mpf, row[5:])))
            amp, phase = scalar.cuts(ring, damp)
            values.append([float(cuts[mode]) for cuts in (amp, phase) for mode in OUTPUT_MODES])
        return np.asarray(values, dtype=np.float64)


def _analytic_jacobian():
    # Output order: four amplitude cuts, then four phase cuts.
    # Input order: five ring frequencies, then five damping frequencies.
    jac = np.zeros((len(ROWS), 8, len(ROWS), 10), dtype=np.float64)
    partials = (
        (0, 1, 0.75), (1, 2, 0.95), (2, 0, 1.0), (3, 4, 0.90),
        (4, 1, 1.0), (4, 6, -1.0),
        (5, 2, 1.0), (5, 7, -1.0),
        (6, 0, 1.0), (6, 5, -0.5),
        (7, 4, 1.0), (7, 9, -1.0),
    )
    for row in range(len(ROWS)):
        for output, input_index, value in partials:
            jac[row, output, row, input_index] = value
    return jac


def test_scalar_default_cuts_and_positive_domain():
    expected = _expected()
    np.testing.assert_allclose(expected[0], [
        0.05625, 0.09025, 0.080, 0.108,
        0.063, 0.080, 0.074, 0.102,
    ], rtol=0, atol=VALUE_ATOL)
    ring = dict(zip(MODES, ROWS[0][:5]))
    damp = dict(zip(MODES, ROWS[0][5:]))
    damp["22"] = "0"
    with pytest.raises(ValueError, match="positive"):
        scalar.cuts(ring, damp)


def test_torch_jax_default_cut_values_and_sparse_jacobian():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import imrphenom_xhm as jax_xhm
    from torchwave.approximants import imrphenom_xhm as torch_xhm

    x = np.asarray(ROWS, dtype=np.float64)
    expected = _expected()
    expected_jac = _analytic_jacobian()

    def torch_evaluate(values):
        ring = {mode: values[:, i] for i, mode in enumerate(MODES)}
        damp = {mode: values[:, i + 5] for i, mode in enumerate(MODES)}
        amp, phase = torch_xhm._ringdown_cut_frequencies(ring, damp)
        return torch.stack([cuts[mode] for cuts in (amp, phase) for mode in OUTPUT_MODES], dim=-1)

    def jax_evaluate(values):
        ring = {mode: values[:, i] for i, mode in enumerate(MODES)}
        damp = {mode: values[:, i + 5] for i, mode in enumerate(MODES)}
        amp, phase = jax_xhm._ringdown_cut_frequencies(ring, damp)
        return jnp.stack([cuts[mode] for cuts in (amp, phase) for mode in OUTPUT_MODES], axis=-1)

    t_input = torch.as_tensor(x, dtype=torch.float64).requires_grad_()
    t_actual = torch_evaluate(t_input).detach().numpy()
    t_jac = torch.autograd.functional.jacobian(torch_evaluate, t_input).detach().numpy()
    j_input = jnp.asarray(x)
    j_actual = np.asarray(jax.jit(jax_evaluate)(j_input))
    j_jac = np.asarray(jax.jit(jax.jacrev(jax_evaluate))(j_input))

    for actual in (t_actual, j_actual):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=VALUE_ATOL)
    for actual in (t_jac, j_jac):
        np.testing.assert_allclose(actual, expected_jac, rtol=0, atol=JAC_ATOL)
