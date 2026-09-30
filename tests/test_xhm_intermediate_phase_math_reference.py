# SPDX-License-Identifier: MIT
"""Published XHM Eq. (38)--(39), Table 7, against both production backends."""

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference import imrphenomxhm_intermediate_phase as ref


CASES = (
    ("0.25", "0.035", "0.100", "0.018", ("100", "86", "71", "42", "15")),
    ("0.10", "0.025", "0.090", "0.014", ("120", "92", "68", "41", "12")),
)


def _scalar(case):
    eta, f_ins, ring, damp = (mp.mpf(x) for x in case[:4])
    targets = tuple(mp.mpf(x) for x in case[4])
    nodes = ref.default_nodes(eta, f_ins, ring)
    coefficients = ref.coefficients(nodes, targets, ring, damp)
    return eta, f_ins, ring, damp, targets, nodes, coefficients


@pytest.mark.parametrize("case", CASES)
def test_scalar_default_collocation_and_integral(case):
    with mp.workdps(70):
        _, _, ring, damp, targets, nodes, coefficients = _scalar(case)
        assert all(left < right for left, right in zip(nodes, nodes[1:]))
        for node, target in zip(nodes, targets):
            assert mp.almosteq(ref.phase_derivative(node, coefficients, ring, damp), target, abs_eps=mp.mpf("1e-55"))
        for frequency in (nodes[0], (nodes[1] + nodes[2]) / 2, nodes[-1]):
            assert mp.almosteq(
                mp.diff(lambda f: ref.phase(f, coefficients, ring, damp), frequency),
                ref.phase_derivative(frequency, coefficients, ring, damp),
                abs_eps=mp.mpf("1e-55"),
            )


@pytest.mark.parametrize("case", CASES)
def test_torch_jax_production_nodes_coefficients_and_phase(case):
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import imrphenom_xhm as jm
    from torchwave.approximants import imrphenom_xhm as tm

    jax.config.update("jax_enable_x64", True)
    with mp.workdps(70):
        eta, f_ins, ring, damp, targets, nodes, coefficients = _scalar(case)
        inputs = tuple(map(float, (eta, f_ins, ring, damp)))
        target_values = np.asarray(list(map(float, targets)))
        tn = tm._nonmix_intermediate_phase_nodes(*(torch.tensor(x, dtype=torch.float64) for x in inputs[:3]))
        jn = jax.jit(jm._nonmix_intermediate_phase_nodes)(*(jnp.asarray(x) for x in inputs[:3]))
        for native_nodes in (np.asarray(tn), np.asarray(jn)):
            np.testing.assert_allclose(native_nodes, list(map(float, nodes)), atol=2e-14, rtol=0)

        tc = tm._nonmix_intermediate_phase_coefficients(
            tn, torch.tensor(target_values), torch.tensor(inputs[2], dtype=torch.float64),
            torch.tensor(inputs[3], dtype=torch.float64),
        )
        jc = jax.jit(jm._nonmix_intermediate_phase_coefficients)(
            jn, jnp.asarray(target_values), jnp.asarray(inputs[2]), jnp.asarray(inputs[3])
        )
        for native_coefficients in (np.asarray(tc), np.asarray(jc)):
            np.testing.assert_allclose(native_coefficients, list(map(float, coefficients)), rtol=5e-8, atol=1e-10)

        frequencies = np.linspace(float(nodes[0]), float(nodes[-1]), 9)
        phase = np.asarray([float(ref.phase(mp.mpf(str(f)), coefficients, ring, damp)) for f in frequencies])
        derivative = np.asarray([
            float(ref.phase_derivative(mp.mpf(str(f)), coefficients, ring, damp)) for f in frequencies
        ])
        tf = torch.tensor(frequencies)
        jf = jnp.asarray(frequencies)
        tr = torch.tensor(inputs[2], dtype=torch.float64)
        td = torch.tensor(inputs[3], dtype=torch.float64)
        jr = jnp.asarray(inputs[2])
        jd = jnp.asarray(inputs[3])
        for got in (
            tm._nonmix_intermediate_phase(tf, tc, tr, td).numpy(),
            np.asarray(jax.jit(jm._nonmix_intermediate_phase)(jf, jc, jr, jd)),
        ):
            np.testing.assert_allclose(got, phase, rtol=0, atol=2e-8)
        for got in (
            tm._nonmix_intermediate_phase_derivative(tf, tc, tr, td).numpy(),
            np.asarray(jax.jit(jm._nonmix_intermediate_phase_derivative)(jf, jc, jr, jd)),
        ):
            np.testing.assert_allclose(got, derivative, rtol=0, atol=2e-8)


def test_target_gradient_through_both_production_solves():
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import imrphenom_xhm as jm
    from torchwave.approximants import imrphenom_xhm as tm

    jax.config.update("jax_enable_x64", True)
    case = CASES[0]
    with mp.workdps(70):
        eta, f_ins, ring, damp, targets, nodes, _ = _scalar(case)
        frequency = mp.mpf("0.067")
        step = mp.mpf("1e-18")

        def scalar_phase(first_target):
            values = (first_target, *targets[1:])
            return ref.phase(frequency, ref.coefficients(nodes, values, ring, damp), ring, damp)

        expected = float((scalar_phase(targets[0] + step) - scalar_phase(targets[0] - step)) / (2 * step))
        tn = tm._nonmix_intermediate_phase_nodes(*(
            torch.tensor(float(x), dtype=torch.float64) for x in (eta, f_ins, ring)
        ))
        tt = torch.tensor(list(map(float, targets)), dtype=torch.float64, requires_grad=True)
        tr = torch.tensor(float(ring), dtype=torch.float64)
        td = torch.tensor(float(damp), dtype=torch.float64)
        tc = tm._nonmix_intermediate_phase_coefficients(tn, tt, tr, td)
        tm._nonmix_intermediate_phase(torch.tensor(float(frequency), dtype=torch.float64), tc, tr, td).backward()

        jn = jm._nonmix_intermediate_phase_nodes(*(jnp.asarray(float(x)) for x in (eta, f_ins, ring)))
        jr, jd = jnp.asarray(float(ring)), jnp.asarray(float(damp))

        def jax_phase(values):
            coeffs = jm._nonmix_intermediate_phase_coefficients(jn, values, jr, jd)
            return jm._nonmix_intermediate_phase(jnp.asarray(float(frequency)), coeffs, jr, jd)
        jgrad = jax.jit(jax.grad(jax_phase))(jnp.asarray(list(map(float, targets))))
        np.testing.assert_allclose((float(tt.grad[0]), float(jgrad[0])), expected, rtol=2e-5, atol=1e-10)


def test_batched_production_path():
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants import imrphenom_xhm as jm
    from torchwave.approximants import imrphenom_xhm as tm

    jax.config.update("jax_enable_x64", True)
    parameters = np.array([[float(x) for x in case[:4]] for case in CASES])
    targets = np.array([[float(x) for x in case[4]] for case in CASES])
    with mp.workdps(70):
        scalar_cases = [_scalar(case) for case in CASES]
        frequencies = np.array([
            np.linspace(float(values[5][0]), float(values[5][-1]), 7)
            for values in scalar_cases
        ])
        expected = np.array([
            [float(ref.phase(mp.mpf(str(f)), values[6], values[2], values[3])) for f in row]
            for values, row in zip(scalar_cases, frequencies)
        ])
    tp = torch.tensor(parameters, dtype=torch.float64)
    tt = torch.tensor(targets, dtype=torch.float64)
    tn = tm._nonmix_intermediate_phase_nodes(tp[:, 0], tp[:, 1], tp[:, 2])
    tc = tm._nonmix_intermediate_phase_coefficients(tn, tt, tp[:, 2], tp[:, 3])
    actual_t = tm._nonmix_intermediate_phase(torch.tensor(frequencies), tc, tp[:, 2], tp[:, 3])
    jp = jnp.asarray(parameters)
    jn = jax.jit(jm._nonmix_intermediate_phase_nodes)(jp[:, 0], jp[:, 1], jp[:, 2])
    jc = jax.jit(jm._nonmix_intermediate_phase_coefficients)(jn, jnp.asarray(targets), jp[:, 2], jp[:, 3])
    actual_j = jax.jit(jm._nonmix_intermediate_phase)(jnp.asarray(frequencies), jc, jp[:, 2], jp[:, 3])
    np.testing.assert_allclose(np.asarray(actual_t), expected, rtol=0, atol=2e-8)
    np.testing.assert_allclose(np.asarray(actual_j), expected, rtol=0, atol=2e-8)
