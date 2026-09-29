# SPDX-License-Identifier: MIT
"""Eq. (38)/(40) and Table-7 (3,2) intermediate-phase component gate."""

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference import imrphenomxhm_transition_mixing as scalar


# Synthetic, positive Mf inputs. The fitted targets and QNM frequencies are
# caller data here; no external implementation supplies the expected results.
CASES = (
    ("0.25", "0.035", "0.09", "0.018", "0.086", "0.10", "0.018", ("21.5", "23.1", "24.0", "24.8")),
    ("0.10", "0.025", "0.082", "0.014", "0.083", "0.09", "0.014", ("17.3", "18.0", "19.1", "20.2")),
)


def _reference(case):
    eta, ins, ring22, damp22, join, ring32, damp32, targets = case
    eta, ins, ring22, damp22, join, ring32, damp32 = map(mp.mpf, (eta, ins, ring22, damp22, join, ring32, damp32))
    targets = tuple(map(mp.mpf, targets))
    nodes = scalar.default_intermediate_32_nodes(eta, ins, ring22, damp22)
    # Synthetic ringdown phase 17f-2f² supplies a join derivative pair.
    dphi_join, d2phi_join = mp.mpf("17.0") - 4 * join, mp.mpf("-4.0")
    coefficients = scalar.intermediate_phase_coefficients(
        nodes, targets, join, dphi_join, d2phi_join, ring32, damp32,
    )
    return (eta, ins, ring22, damp22, join, ring32, damp32, targets,
            nodes, dphi_join, d2phi_join, coefficients)


@pytest.mark.parametrize("case", CASES)
def test_scalar_table7_and_six_eq40_conditions(case):
    with mp.workdps(70):
        _, _, _, _, join, ring32, damp32, targets, nodes, dphi, d2phi, coeffs = _reference(case)
        assert nodes[0] < nodes[1] < nodes[2] < nodes[3] < join
        for node, target in zip(nodes, targets):
            assert abs(scalar.intermediate_phase_derivative(node, ring32, damp32, coeffs) - target) < mp.mpf("1e-50")
        assert abs(scalar.intermediate_phase_derivative(join, ring32, damp32, coeffs) - dphi) < mp.mpf("1e-50")
        assert abs(scalar.intermediate_phase_second_derivative(join, ring32, damp32, coeffs) - d2phi) < mp.mpf("1e-48")
        assert abs(mp.diff(lambda f: scalar.intermediate_phase(f, ring32, damp32, coeffs), join)
                   - dphi) < mp.mpf("1e-48")


@pytest.mark.parametrize("case", CASES)
def test_production_torch_jax_values_and_target_gradients(case):
    import jax
    import jax.numpy as jnp
    import torch
    jax.config.update("jax_enable_x64", True)
    from jaxwave.approximants import imrphenom_xhm as jax_xhm
    from torchwave.approximants import imrphenom_xhm as torch_xhm

    with mp.workdps(70):
        (eta, ins, ring22, damp22, join, ring32, damp32, targets,
         nodes, dphi, d2phi, coeffs) = _reference(case)
        freq = (nodes[0], (nodes[1] + nodes[2]) / 2, nodes[3], join)
        expected_phase = [float(scalar.intermediate_phase(f, ring32, damp32, coeffs)) for f in freq]
        expected_derivative = [float(scalar.intermediate_phase_derivative(f, ring32, damp32, coeffs)) for f in freq]
        h = mp.mpf("1e-18")

        def target_phase(t0):
            t = (t0, *targets[1:])
            c = scalar.intermediate_phase_coefficients(nodes, t, join, dphi, d2phi, ring32, damp32)
            return scalar.intermediate_phase(freq[1], ring32, damp32, c)

        expected_gradient = float((target_phase(targets[0] + h) - target_phase(targets[0] - h)) / (2 * h))

    def t(value):
        return torch.tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)

    args = (eta, ins, ring22, damp22)
    t_nodes = torch_xhm._intermediate_32_phase_nodes(*(t(x) for x in args))
    j_nodes = jax.jit(jax_xhm._intermediate_32_phase_nodes)(*(jnp.asarray(float(x)) for x in args))
    for actual in (t_nodes.detach().numpy(), np.asarray(j_nodes)):
        np.testing.assert_allclose(actual, np.asarray([float(x) for x in nodes]), rtol=0, atol=2e-14)

    t_targets = t([float(x) for x in targets]).requires_grad_()
    t_coeffs = torch_xhm._intermediate_32_phase_coefficients(
        t_nodes, t_targets, t(join), t(dphi), t(d2phi), t(ring32), t(damp32),
    )

    def j_solve(target):
        return jax_xhm._intermediate_32_phase_coefficients(
            j_nodes, target, jnp.asarray(float(join)), jnp.asarray(float(dphi)),
            jnp.asarray(float(d2phi)), jnp.asarray(float(ring32)), jnp.asarray(float(damp32)),
        )

    j_targets = jnp.asarray([float(x) for x in targets])
    j_coeffs = jax.jit(j_solve)(j_targets)
    for actual in (t_coeffs.detach().numpy(), np.asarray(j_coeffs)):
        np.testing.assert_allclose(actual, [float(x) for x in coeffs], rtol=5e-7, atol=1e-10)

    t_phase = torch_xhm._intermediate_32_phase(t([float(x) for x in freq]), t_coeffs, t(ring32), t(damp32))
    t_deriv = torch_xhm._intermediate_32_phase_derivative(t([float(x) for x in freq]), t_coeffs, t(ring32), t(damp32))
    j_phase = jax.jit(jax_xhm._intermediate_32_phase)(jnp.asarray([float(x) for x in freq]), j_coeffs,
                                                      jnp.asarray(float(ring32)), jnp.asarray(float(damp32)))
    j_deriv = jax.jit(jax_xhm._intermediate_32_phase_derivative)(jnp.asarray([float(x) for x in freq]), j_coeffs,
                                                                 jnp.asarray(float(ring32)), jnp.asarray(float(damp32)))
    for actual in (t_phase.detach().numpy(), np.asarray(j_phase)):
        np.testing.assert_allclose(actual, expected_phase, rtol=0, atol=2e-7)
    for actual in (t_deriv.detach().numpy(), np.asarray(j_deriv)):
        np.testing.assert_allclose(actual, expected_derivative, rtol=0, atol=2e-7)

    t_grad = torch.autograd.grad(torch_xhm._intermediate_32_phase(t(freq[1]), t_coeffs, t(ring32), t(damp32)), t_targets)[0][0]

    def j_target_phase(target):
        return jax_xhm._intermediate_32_phase(jnp.asarray(float(freq[1])), j_solve(target),
                                               jnp.asarray(float(ring32)), jnp.asarray(float(damp32)))

    j_grad = jax.jit(jax.grad(j_target_phase))(j_targets)[0]
    for actual in (float(t_grad), float(j_grad)):
        np.testing.assert_allclose(actual, expected_gradient, rtol=2e-4, atol=1e-9)

    t_f = t(join).requires_grad_()
    t_second = torch.autograd.grad(torch_xhm._intermediate_32_phase_derivative(t_f, t_coeffs, t(ring32), t(damp32)), t_f)[0]
    j_second = jax.jit(jax.grad(lambda f: jax_xhm._intermediate_32_phase_derivative(
        f, j_coeffs, jnp.asarray(float(ring32)), jnp.asarray(float(damp32)))))(jnp.asarray(float(join)))
    for actual in (float(t_second), float(j_second)):
        np.testing.assert_allclose(actual, float(d2phi), rtol=0, atol=2e-5)


def test_production_batched_six_condition_solve():
    import jax
    import jax.numpy as jnp
    import torch
    jax.config.update("jax_enable_x64", True)
    from jaxwave.approximants import imrphenom_xhm as jax_xhm
    from torchwave.approximants import imrphenom_xhm as torch_xhm

    with mp.workdps(70):
        expected = [_reference(case) for case in CASES]

    def columns(*indices):
        return np.asarray([[float(row[i]) for i in indices] for row in expected], dtype=np.float64)

    eta, ins, ring22, damp22 = (columns(i).ravel() for i in range(4))
    join, ring32, damp32 = (columns(i).ravel() for i in (4, 5, 6))
    targets = np.asarray(
        [[float(x) for x in row[7]] for row in expected], dtype=np.float64,
    )
    first, second = (columns(i).ravel() for i in (9, 10))
    expected_coeffs = np.asarray([[float(x) for x in row[11]] for row in expected])

    tn = torch_xhm._intermediate_32_phase_nodes(*(torch.as_tensor(x) for x in (eta, ins, ring22, damp22)))
    tc = torch_xhm._intermediate_32_phase_coefficients(
        tn, torch.as_tensor(targets), torch.as_tensor(join), torch.as_tensor(first),
        torch.as_tensor(second), torch.as_tensor(ring32), torch.as_tensor(damp32),
    )
    jn = jax.jit(jax_xhm._intermediate_32_phase_nodes)(*(jnp.asarray(x) for x in (eta, ins, ring22, damp22)))
    jc = jax.jit(jax_xhm._intermediate_32_phase_coefficients)(
        jn, jnp.asarray(targets), jnp.asarray(join), jnp.asarray(first),
        jnp.asarray(second), jnp.asarray(ring32), jnp.asarray(damp32),
    )
    for actual in (tc.detach().numpy(), np.asarray(jc)):
        np.testing.assert_allclose(actual, expected_coeffs, rtol=5e-7, atol=1e-10)
