# SPDX-License-Identifier: MIT
"""Independent checks of printed XHM (3,2) transition and mode mixing."""

import mpmath as mp
import numpy as np

from diffgw_reference import imrphenomxhm_transition_mixing as ref


RING = 0.10
DAMP = 0.018
JOIN = 0.086
NODES = (0.046, 0.056, 0.066, 0.076)
ALPHAS = (1.05 + 0.02j, 0.04 - 0.01j, -0.08 + 0.03j, 0.96 - 0.02j)


def _h22(f):
    return (1 + 0.3 * f) * mp.exp(-1j * (12 * f + 2 * f**2))


def _h32_s(f):
    return (0.37 + 0.2 * f) * mp.exp(-1j * (18 * f - f**2))


def _mixed(f):
    return ref.spherical_32_from_spheroidal(_h22(f), _h32_s(f), *ALPHAS)


def _transition_data():
    with mp.workdps(70):
        phase, dphase, d2phase = ref.mixed_phase_derivatives(JOIN, _h22, _h32_s, ALPHAS)
        targets = tuple(mp.mpf(x) for x in (21.5, 23.1, 24.0, 24.8))
        coeffs = ref.intermediate_phase_coefficients(NODES, targets, JOIN, dphase, d2phase, RING, DAMP)
        offset = phase - ref.intermediate_phase(JOIN, RING, DAMP, coeffs)
        return tuple(map(float, targets)), float(dphase), float(d2phase), coeffs, offset


def test_scalar_mixing_eliminates_spheroidal_22_and_has_phase_derivatives():
    with mp.workdps(70):
        f = mp.mpf("0.086")
        h22, h32_s = _h22(f), _h32_s(f)
        a222, a232, a322, a332 = ALPHAS
        h22_s = (h22 - a232 * h32_s) / a222
        np.testing.assert_allclose(complex(_mixed(f)), complex(a322 * h22_s + a332 * h32_s), rtol=2e-15)
        phase, first, second = ref.mixed_phase_derivatives(f, _h22, _h32_s, ALPHAS)
        np.testing.assert_allclose(float(first), float(mp.diff(lambda x: -mp.arg(_mixed(x)), f)), rtol=2e-14)
        np.testing.assert_allclose(float(second), float(mp.diff(lambda x: -mp.arg(_mixed(x)), f, 2)), rtol=2e-13)
        np.testing.assert_allclose(float(phase), float(-mp.arg(_mixed(f))), rtol=2e-15)


def test_scalar_intermediate_collocation_and_c2_join():
    targets, dphase, d2phase, coeffs, offset = _transition_data()
    with mp.workdps(70):
        for node, target in zip(NODES, targets):
            assert mp.almosteq(ref.intermediate_phase_derivative(node, RING, DAMP, coeffs), target, rel_eps=mp.mpf("1e-48"))
        def phase(f):
            return ref.intermediate_phase(f, RING, DAMP, coeffs, offset=offset)

        def mixed_phase(f):
            return -mp.arg(_mixed(f))
        for order in (0, 1, 2):
            np.testing.assert_allclose(float(mp.diff(phase, JOIN, order)), float(mp.diff(mixed_phase, JOIN, order)), rtol=3e-11, atol=3e-11)
        spheroidal = (0.2, 0.0001, 0.3, 8.0)
        np.testing.assert_allclose(
            float(mp.diff(lambda f: ref.ringdown_spheroidal_phase(f, RING, DAMP, spheroidal), JOIN)),
            float(ref.ringdown_spheroidal_phase_derivative(JOIN, RING, DAMP, spheroidal)),
            rtol=2e-14,
        )


def test_production_torch_jax_mixing_and_transition_solve():
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants.imrphenom_xhm import (
        _intermediate_32_phase_coefficients as jax_solve,
        _spherical_32_from_spheroidal as jax_mix,
    )
    from torchwave.approximants.imrphenom_xhm import (
        _intermediate_32_phase_coefficients as torch_solve,
        _spherical_32_from_spheroidal as torch_mix,
        _spherical_32_from_spheroidal_cartesian as torch_mix_cartesian,
    )

    jax.config.update("jax_enable_x64", True)
    targets, dphase, d2phase, coeffs, _ = _transition_data()
    for f in (0.049, JOIN, 0.097):
        expected = complex(_mixed(mp.mpf(f)))
        h22, h32_s = complex(_h22(mp.mpf(f))), complex(_h32_s(mp.mpf(f)))
        def tc(x):
            return torch.tensor(x, dtype=torch.complex128)
        got_t = torch_mix(tc(h22), tc(h32_s), *(tc(x) for x in ALPHAS))
        got_j = jax_mix(jnp.asarray(h22), jnp.asarray(h32_s), *(jnp.asarray(x) for x in ALPHAS))
        ratio = ALPHAS[2] / ALPHAS[0]
        carry = ALPHAS[3] - ratio * ALPHAS[1]
        got_re, got_im = torch_mix_cartesian(
            torch.tensor(h22.real, dtype=torch.float64), torch.tensor(h22.imag, dtype=torch.float64),
            torch.tensor(h32_s.real, dtype=torch.float64),
            *(torch.tensor(x, dtype=torch.float64) for x in (ratio.real, ratio.imag, carry.real, carry.imag)),
        )
        np.testing.assert_allclose((complex(got_t.item()), complex(got_j)), expected, rtol=2e-14)
        np.testing.assert_allclose(complex(got_re.item(), got_im.item()), ratio * h22 + carry * h32_s.real, rtol=2e-14)

    inputs = (NODES, targets, JOIN, dphase, d2phase, RING, DAMP)
    def t(x):
        return torch.tensor(x, dtype=torch.float64)
    result_t = torch_solve(*(t(x) for x in inputs)).detach().numpy()
    result_j = np.asarray(jax.jit(jax_solve)(*(jnp.asarray(x) for x in inputs)))
    for result in (result_t, result_j):
        for f in (*NODES, JOIN, 0.082):
            expected = float(ref.intermediate_phase_derivative(f, RING, DAMP, coeffs))
            got = float(ref.intermediate_phase_derivative(f, RING, DAMP, result))
            np.testing.assert_allclose(got, expected, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(float(mp.diff(lambda x: ref.intermediate_phase_derivative(x, RING, DAMP, result), JOIN)), d2phase, rtol=2e-6, atol=2e-6)


def test_production_mixed_phase_first_and_second_derivatives():
    import jax
    import jax.numpy as jnp
    import torch
    from jaxwave.approximants.imrphenom_xhm import _spherical_32_from_spheroidal as jax_mix
    from torchwave.approximants.imrphenom_xhm import _spherical_32_from_spheroidal as torch_mix

    jax.config.update("jax_enable_x64", True)
    _, expected_first, expected_second = ref.mixed_phase_derivatives(JOIN, _h22, _h32_s, ALPHAS)

    def torch_phase(f):
        h22 = (1 + 0.3 * f) * torch.exp(-1j * (12 * f + 2 * f**2))
        h32_s = (0.37 + 0.2 * f) * torch.exp(-1j * (18 * f - f**2))
        alphas = (torch.tensor(x, dtype=torch.complex128) for x in ALPHAS)
        return -torch.angle(torch_mix(h22, h32_s, *alphas))

    def jax_phase(f):
        h22 = (1 + 0.3 * f) * jnp.exp(-1j * (12 * f + 2 * f**2))
        h32_s = (0.37 + 0.2 * f) * jnp.exp(-1j * (18 * f - f**2))
        return -jnp.angle(jax_mix(h22, h32_s, *(jnp.asarray(x) for x in ALPHAS)))

    t = torch.tensor(JOIN, dtype=torch.float64)
    tangent_t = torch.ones_like(t)

    def torch_first(x):
        return torch.func.jvp(torch_phase, (x,), (torch.ones_like(x),))[1]

    first_t = torch_first(t)
    second_t = torch.func.jvp(torch_first, (t,), (tangent_t,))[1]

    j = jnp.asarray(JOIN)

    def jax_first(x):
        return jax.jvp(jax_phase, (x,), (jnp.ones_like(x),))[1]

    first_j = jax_first(j)
    second_j = jax.jvp(jax_first, (j,), (jnp.ones_like(j),))[1]
    np.testing.assert_allclose((float(first_t), float(first_j)), float(expected_first), rtol=3e-12)
    np.testing.assert_allclose((float(second_t), float(second_j)), float(expected_second), rtol=3e-11)
