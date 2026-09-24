"""Nonpublic JAX aligned v223 geometry and Eq. (45) projection checks.

The observer basis and quadrupole rotation follow Pratten et al.,
arXiv:2004.06503v2, Appendix C and Eq. (45).  The finite beta and aligned
chart are v223 compatibility prescriptions; these tests do not gate public
dispatch or the near-aligned transverse-spin limit.
"""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxwave.approximants.precession_msa_v223_projection import (
    aligned_msa_v223_waveform,
    aligned_source_frame_geometry_v223,
    project_l2_eq45,
)
from torchwave.approximants.precession_msa_angles_clean import (
    assemble_precession_msa_angles_v223,
)
from torchwave.approximants.precession_msa_waveform_clean import (
    _native_msa_final_spin,
    waveform_v223,
)
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry
from torchwave.approximants.precession_twisting_clean import (
    project_twisted_positive_modes,
)


def _t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def _relative_l2(actual, target):
    return np.linalg.norm(np.asarray(actual) - np.asarray(target)) / np.linalg.norm(target)


@pytest.mark.parametrize("phi_ref", (0.0, 0.3, 0.7))
def test_aligned_source_geometry_matches_appendix_c_basis(phi_ref):
    jax_geo = aligned_source_frame_geometry_v223(
        40.0, 20.0, 0.4, -0.2, 0.4, phi_ref, 40.0,
    )
    torch_geo = source_frame_geometry(
        40.0, 20.0, 0.0, 0.0, 0.4, 0.0, 0.0, -0.2,
        0.4, phi_ref, 40.0, compatibility="v223",
    )
    for key, value in zip(("alpha0", "theta_jn", "phi_j"),
                          (torch_geo[0], torch_geo[1], torch_geo[3])):
        np.testing.assert_allclose(np.asarray(jax_geo[key]), value.detach().numpy(), atol=2e-15)
    # zeta=+pi and -pi describe the same spin-2 observer basis.
    np.testing.assert_allclose(
        np.exp(2j * np.asarray(jax_geo["zeta"])),
        np.exp(2j * np.asarray(torch_geo[2].detach())), atol=2e-15,
    )
    np.testing.assert_allclose(
        np.asarray(jax_geo["phi_j"]), np.pi / 2.0 - phi_ref, atol=2e-15,
    )


def test_finite_beta_eq45_projection_matches_independent_torch_sum():
    hneg = np.array([2.0 + 1.0j, -0.3 + 0.8j, 1.2 - 0.2j])
    alpha = np.array([0.2, -0.5, 1.1])
    gamma = np.array([0.8, 0.3, -0.1])
    beta = np.array([0.0, 0.4, 1.1])
    theta, zeta = 0.7, -0.3
    jhp, jhc = project_l2_eq45(
        jnp.asarray(hneg), jnp.asarray(alpha), jnp.asarray(gamma),
        jnp.cos(jnp.asarray(beta)), theta, zeta,
    )
    thp, thc = project_twisted_positive_modes(
        {(2, 2): torch.as_tensor(hneg)}, _t(alpha), _t(gamma), _t(beta),
        _t(theta), _t(zeta),
    )
    np.testing.assert_allclose(np.asarray(jhp), thp.detach().numpy(), rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(np.asarray(jhc), thc.detach().numpy(), rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("phi_ref,convention", [
    (0.0, 0), (0.3, 0), (0.3, 1), (0.7, 1),
])
def test_aligned_waveform_and_intermediates_match_torch_clean(phi_ref, convention):
    f = np.array([25.0, 40.0, 80.0])
    jhp, jhc, state = aligned_msa_v223_waveform(
        jnp.asarray(f), 40.0, 20.0, 0.4, -0.2,
        inclination=0.4, phic=phi_ref, f_ref=40.0,
        convention=convention, return_state=True,
    )
    thp, thc = waveform_v223(
        _t(f), 40.0, 20.0, spin1z=0.4, spin2z=-0.2,
        inclination=0.4, phic=phi_ref, f_ref=40.0,
        convention=convention,
    )
    assert _relative_l2(jhp, thp.detach().numpy()) < 5e-13
    assert _relative_l2(jhc, thc.detach().numpy()) < 5e-13
    spin1, spin2 = _t([0.0, 0.0, 0.4]), _t([0.0, 0.0, -0.2])
    angles = assemble_precession_msa_angles_v223(
        _t(f), _t(40.0), _t(20.0), spin1, spin2, f_ref=_t(40.0),
    )
    remnant, _ = _native_msa_final_spin(
        _t(40.0), _t(20.0), spin1, spin2, _t(40.0), angles,
        compatibility="v223",
    )
    phi_j = np.pi / 2.0 - phi_ref
    expected_alpha = angles["alpha"].detach().numpy() + np.pi
    expected_gamma = angles["gamma"].detach().numpy()
    if convention == 1:
        expected_alpha -= phi_j
        expected_gamma += np.pi
    np.testing.assert_allclose(np.asarray(state["alpha"]), expected_alpha, atol=5e-10)
    np.testing.assert_allclose(np.asarray(state["gamma"]), expected_gamma, atol=5e-10)
    np.testing.assert_allclose(np.asarray(state["cos_beta"]), angles["cos_beta"].detach().numpy(), atol=2e-14)
    np.testing.assert_allclose(np.asarray(state["final_spin"]), remnant.detach().numpy(), atol=2e-13)
    np.testing.assert_allclose(np.asarray(state["theta_jn"]), 0.4, atol=2e-15)
    np.testing.assert_allclose(np.asarray(state["phi_j"]), np.pi / 2.0 - phi_ref, atol=2e-15)
    assert np.all(1.0 - np.asarray(state["cos_beta"]) > 0.0)
    assert np.all(1.0 - np.asarray(state["cos_beta"]) < 1e-8)


def test_aligned_waveform_batched_observer_parameters_match_rows():
    f = jnp.asarray([25.0, 40.0, 80.0])
    rows = ((40.0, 20.0, 0.4, -0.2, 0.4, 0.0, 40.0),
            (35.0, 25.0, 0.2, -0.1, 0.7, 0.3, 20.0))
    hp, hc = aligned_msa_v223_waveform(
        f,
        *[jnp.asarray([row[index] for row in rows]) for index in range(4)],
        inclination=jnp.asarray([row[4] for row in rows]),
        phic=jnp.asarray([row[5] for row in rows]),
        f_ref=jnp.asarray([row[6] for row in rows]),
    )
    assert hp.shape == hc.shape == (2, 3)
    for index, (m1, m2, chi1, chi2, inc, phi, fr) in enumerate(rows):
        one_hp, one_hc = aligned_msa_v223_waveform(
            f, m1, m2, chi1, chi2, inclination=inc, phic=phi, f_ref=fr,
        )
        np.testing.assert_allclose(np.asarray(hp[index]), np.asarray(one_hp), rtol=2e-13)
        np.testing.assert_allclose(np.asarray(hc[index]), np.asarray(one_hc), rtol=2e-13)


@pytest.mark.parametrize("convention", (0, 1))
def test_separate_observer_azimuth_obeys_aligned_spin2_identity(convention):
    f = jnp.asarray([25.0, 40.0, 80.0])

    def waveform(phi0):
        return aligned_msa_v223_waveform(
            f, 40.0, 20.0, 0.4, -0.2,
            inclination=0.4, phic=0.3, observer_azimuth=phi0,
            f_ref=40.0, convention=convention,
        )

    compiled = jax.jit(waveform)
    base = compiled(jnp.asarray(0.0))
    for phi0 in (0.4, -0.7):
        shifted = compiled(jnp.asarray(phi0))
        for actual, unrotated in zip(shifted, base):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.exp(-2.0j * phi0) * np.asarray(unrotated),
                rtol=3e-14, atol=0.0,
            )
        values, tangents = jax.jvp(waveform, (jnp.asarray(phi0),), (jnp.asarray(1.0),))
        for actual, value in zip(tangents, values):
            np.testing.assert_allclose(
                np.asarray(actual), -2.0j * np.asarray(value),
                rtol=3e-14, atol=0.0,
            )


@pytest.mark.parametrize("convention", (0, 1))
def test_aligned_waveform_phic_gradient_matches_torch_clean(convention):
    f = jnp.asarray([25.0, 40.0, 80.0])

    def jax_loss(phi):
        hp, hc = aligned_msa_v223_waveform(
            f, 40.0, 20.0, 0.4, -0.2,
            inclination=0.4, phic=phi, f_ref=40.0,
            convention=convention,
        )
        return jnp.real(hp[0] * 1e22) + jnp.imag(hc[2] * 1e22)

    phi = _t(0.3).requires_grad_()
    thp, thc = waveform_v223(
        _t([25.0, 40.0, 80.0]), 40.0, 20.0,
        spin1z=0.4, spin2z=-0.2, inclination=0.4,
        phic=phi, f_ref=40.0, convention=convention,
    )
    torch_loss = (thp[0] * 1e22).real + (thc[2] * 1e22).imag
    torch_grad = torch.autograd.grad(torch_loss, phi)[0]
    jax_grad = jax.grad(jax_loss)(jnp.asarray(0.3))
    assert bool(jnp.isfinite(jax_grad))
    np.testing.assert_allclose(np.asarray(jax_grad), torch_grad.detach().numpy(), rtol=2e-10)


@pytest.mark.parametrize("convention", (0, 1))
def test_aligned_dense_waveform_matches_public_v223_compatibility_oracle(convention):
    lal = pytest.importorskip("lal")
    ls = pytest.importorskip("lalsimulation")
    frequency = jnp.arange(20.0, 512.0, 0.25)
    hp, hc = aligned_msa_v223_waveform(
        frequency, 35.0, 25.0, 0.2, -0.1,
        distance=500.0, inclination=0.7, phic=0.3,
        f_ref=20.0, convention=convention,
    )
    options = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(options, 223)
    ls.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(options, 5)
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(options, convention)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(options, 3)
    target_hp, target_hc = ls.SimInspiralChooseFDWaveform(
        35.0 * lal.MSUN_SI, 25.0 * lal.MSUN_SI,
        0.0, 0.0, 0.2, 0.0, 0.0, -0.1,
        500e6 * lal.PC_SI, 0.7, 0.3, 0.0, 0.0, 0.0,
        0.25, 20.0, 512.0, 20.0, options, ls.IMRPhenomXP,
    )
    for actual, target in ((hp, target_hp), (hc, target_hc)):
        reference = np.asarray(target.data.data[80:2048])
        assert _relative_l2(actual, reference) < 1e-8
