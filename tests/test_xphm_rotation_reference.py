"""Independent scalar checks for the IMRPhenomXPHM twist-up layer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from diffgw_core.xphm_options import resolve_xphm_options
from diffgw_reference.imrphenomxphm_rotation import (
    polarizations_from_primed,
    polarizations_to_primed,
    rotate_polarizations,
    spa_twist_frequency,
    twist_up_mode as scalar_twist_up_mode,
    wigner_small_d,
)
from jaxwave.approximants.imrphenom_xphm import (
    IMRPhenomXPHM as JaxXPHM,
    twist_up_mode as jax_twist_up_mode,
)
from torchwave.approximants.imrphenom_xphm import (
    IMRPhenomXPHM as TorchXPHM,
    twist_up_mode as torch_twist_up_mode,
    wigner_small_d_matrix,
)

jax.config.update("jax_enable_x64", True)
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))


@pytest.mark.parametrize("l", (2, 3, 4))
def test_scalar_wigner_matrix_matches_tensor_kernel(l):
    beta = 0.731
    actual = wigner_small_d_matrix(l, torch.tensor(beta, dtype=torch.float64)).numpy()
    expected = np.array(
        [[wigner_small_d(l, m, mp, beta) for mp in range(-l, l + 1)] for m in range(-l, l + 1)]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(expected @ expected.T, np.eye(2 * l + 1), atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("mode", MODES)
def test_scalar_twist_up_matches_torch_and_jax(mode):
    l, mp = mode
    carrier = complex(-0.37, 0.29)
    angles = dict(alpha=0.43, beta=1.07, gamma=-0.31, theta_jn=0.82, zeta=0.19)
    hp_ref, hc_ref = scalar_twist_up_mode(l, mp, carrier, **angles)

    def tt(value):
        return torch.tensor(value, dtype=torch.float64)

    hp_t, hc_t = torch_twist_up_mode(
        l, mp, torch.tensor(carrier, dtype=torch.complex128), **{key: tt(value) for key, value in angles.items()}
    )
    hp_j, hc_j = jax_twist_up_mode(
        l,
        mp,
        jnp.asarray(carrier, dtype=jnp.complex128),
        **{key: jnp.asarray(value, dtype=jnp.float64) for key, value in angles.items()},
    )
    np.testing.assert_allclose(complex(hp_t), hp_ref, atol=3e-15, rtol=3e-14)
    np.testing.assert_allclose(complex(hc_t), hc_ref, atol=3e-15, rtol=3e-14)
    np.testing.assert_allclose(complex(hp_j), hp_ref, atol=3e-15, rtol=3e-14)
    np.testing.assert_allclose(complex(hc_j), hc_ref, atol=3e-15, rtol=3e-14)


def test_nnlo_convention_one_sign_does_not_repeat_xhm_mode_adapter():
    # Pratten et al. Eqs. (110)--(113) fix the twist-up sum, while Appendix F
    # Table IV supplies the convention-1 offsets. After the shared XHM
    # adapter i**mp, the remaining convention-1 carrier correction is only a
    # common -1. Choosing odd mp makes a second i**mp detectably non-real.
    l, mp = 3, 3
    carrier = complex(-0.37, 0.29)
    angles = dict(alpha=0.43, beta=1.07, gamma=-0.31, theta_jn=0.82, zeta=0.19)
    base = scalar_twist_up_mode(l, mp, carrier, **angles)
    expected = tuple(-value for value in base)
    scalar_signed = scalar_twist_up_mode(l, mp, carrier, **angles, carrier_sign=-1.0)

    def tt(value):
        return torch.tensor(value, dtype=torch.float64)

    torch_signed = torch_twist_up_mode(
        l,
        mp,
        torch.tensor(carrier, dtype=torch.complex128),
        **{key: tt(value) for key, value in angles.items()},
        carrier_sign=-1.0,
    )
    jax_signed = jax_twist_up_mode(
        l,
        mp,
        jnp.asarray(carrier, dtype=jnp.complex128),
        **{key: jnp.asarray(value, dtype=jnp.float64) for key, value in angles.items()},
        carrier_sign=-1.0,
    )
    for actual in (scalar_signed, tuple(complex(x) for x in torch_signed), tuple(complex(x) for x in jax_signed)):
        np.testing.assert_allclose(actual, expected, atol=3e-15, rtol=3e-14)
        wrong_double_adapter = tuple((1j**mp) * value for value in expected)
        assert not np.allclose(actual, wrong_double_adapter, atol=1e-12, rtol=1e-12)


def test_polarization_rotation_is_norm_preserving_and_composes():
    hp, hc = complex(0.2, -0.7), complex(-0.4, 0.3)
    zeta1, zeta2 = 0.23, -0.61
    once = rotate_polarizations(hp, hc, zeta1 + zeta2)
    twice = rotate_polarizations(*rotate_polarizations(hp, hc, zeta1), zeta2)
    np.testing.assert_allclose(twice, once, atol=3e-16, rtol=3e-15)
    assert abs(sum(abs(x) ** 2 for x in once) - (abs(hp) ** 2 + abs(hc) ** 2)) < 3e-16


def test_paper_polarization_forward_and_inverse_signs():
    hp, hc = complex(0.2, -0.7), complex(-0.4, 0.3)
    zeta = np.pi / 4.0

    # At 2*zeta=pi/2, Eqs. (93)--(94) require (hp', hc')=(-hc, hp).
    primed = polarizations_to_primed(hp, hc, zeta)
    np.testing.assert_allclose(primed, (-hc, hp), atol=2e-16, rtol=2e-15)

    # The XPHM output step is the inverse map: (hp, hc)=(hc', -hp').
    inverse_at_same_inputs = polarizations_from_primed(hp, hc, zeta)
    np.testing.assert_allclose(inverse_at_same_inputs, (hc, -hp), atol=2e-16, rtol=2e-15)
    np.testing.assert_allclose(
        polarizations_from_primed(*primed, zeta),
        (hp, hc),
        atol=2e-16,
        rtol=2e-15,
    )


@pytest.mark.parametrize("mp, expected", ((1, 160.0), (2, 80.0), (3, 160.0 / 3.0), (4, 40.0)))
def test_per_mode_spa_frequency(mp, expected):
    assert spa_twist_frequency(80.0, mp) == expected


@pytest.mark.parametrize(
    "method, version, final_spin, expected",
    (
        ("msa", None, None, ("msa", 223, 3)),
        ("nnlo", None, None, ("nnlo", 102, 0)),
        ("nnlo", 224, None, ("msa", 224, 3)),
        ("spintaylor", 101, None, ("nnlo", 101, 0)),
        ("pnr", 223, 0, ("msa", 223, 0)),
        ("st", None, None, ("spintaylor", 320, 2)),
        ("xpnr", 300, None, ("pnr", 330, 2)),
    ),
)
def test_option_resolution_is_version_led(method, version, final_spin, expected):
    assert resolve_xphm_options(method, version, final_spin) == expected


@pytest.mark.parametrize(
    "method, version, final_spin, message",
    (
        ("ode", None, None, "unvalidated prototype"),
        ("unknown", None, None, "precession_method must"),
        (None, None, None, "precession_method must"),
        ("msa", True, None, "Unsupported precession version"),
        ("msa", 223.0, None, "Unsupported precession version"),
        ("msa", 999, None, "Unsupported precession version"),
        ("msa", None, True, "final_spin_mod must"),
        ("msa", None, 3.0, "final_spin_mod must"),
    ),
)
def test_option_resolution_rejects_ambiguous_or_unsupported_requests(
    method, version, final_spin, message
):
    with pytest.raises(ValueError, match=message):
        resolve_xphm_options(method, version, final_spin)


def _waveform_kwargs():
    return dict(
        mass1=35.0,
        mass2=25.0,
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.2,
        spin2x=-0.1,
        spin2y=0.15,
        spin2z=-0.1,
        distance=500.0,
        inclination=0.7,
        phic=0.3,
        f_ref=20.0,
    )


def test_public_defaults_are_msa_223_final_spin_3_in_both_backends():
    assert TorchXPHM().default_precession_method == "msa"
    assert JaxXPHM().default_precession_method == "msa"
    f_t = torch.tensor([25.0, 40.0, 75.0], dtype=torch.float64)
    f_j = jnp.asarray([25.0, 40.0, 75.0], dtype=jnp.float64)
    kwargs = _waveform_kwargs()
    implicit_t = TorchXPHM(modes=[(2, 2)])(f=f_t, **kwargs)
    explicit_t = TorchXPHM(modes=[(2, 2)])(
        f=f_t, precession_method="msa", prec_version=223, final_spin_mod=3, **kwargs
    )
    implicit_j = JaxXPHM(modes=[(2, 2)])(f=f_j, **kwargs)
    explicit_j = JaxXPHM(modes=[(2, 2)])(
        f=f_j, precession_method="msa", prec_version=223, final_spin_mod=3, **kwargs
    )
    for implicit, explicit in zip(implicit_t, explicit_t):
        torch.testing.assert_close(implicit, explicit, atol=0.0, rtol=0.0)
    for implicit, explicit in zip(implicit_j, explicit_j):
        np.testing.assert_array_equal(np.asarray(implicit), np.asarray(explicit))


def test_explicit_version_overrides_conflicting_method_in_both_backends():
    f_t = torch.tensor([25.0, 40.0], dtype=torch.float64)
    f_j = jnp.asarray([25.0, 40.0], dtype=jnp.float64)
    kwargs = _waveform_kwargs()
    conflicting_t = TorchXPHM(modes=[(2, 2)])(
        f=f_t, precession_method="spintaylor", prec_version=223, **kwargs
    )
    canonical_t = TorchXPHM(modes=[(2, 2)])(
        f=f_t, precession_method="msa", prec_version=223, **kwargs
    )
    conflicting_j = JaxXPHM(modes=[(2, 2)])(
        f=f_j, precession_method="spintaylor", prec_version=223, **kwargs
    )
    canonical_j = JaxXPHM(modes=[(2, 2)])(
        f=f_j, precession_method="msa", prec_version=223, **kwargs
    )
    for conflicting, canonical in zip(conflicting_t, canonical_t):
        torch.testing.assert_close(conflicting, canonical, atol=0.0, rtol=0.0)
    for conflicting, canonical in zip(conflicting_j, canonical_j):
        np.testing.assert_array_equal(np.asarray(conflicting), np.asarray(canonical))
