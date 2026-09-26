"""Appendix-C source-frame transport gates independent of waveform frontends."""

from __future__ import annotations

import mpmath
import numpy as np
import pytest

from diffgw_reference.xp_msa_generic_boundary import source_chart, source_chart_transport


def _scalar_charts(orbital_l, spin, inclination, phi_ref):
    initial = source_chart(spin, orbital_l=orbital_l, inclination=inclination,
                           phi_ref="0")
    final = source_chart(spin, orbital_l=orbital_l, inclination=inclination,
                         phi_ref=str(phi_ref))
    return initial, final, source_chart_transport(initial, final)


@pytest.mark.parametrize("phi_ref", ("0.3", "-0.1", "3.4", "6.583185307179586"))
def test_torch_and_jax_v223_geometry_follow_scalar_appendix_c(phi_ref):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from jaxwave.approximants.precession_msa_v223_generic_projection import (
        generic_source_frame_geometry_v223,
    )
    from torchwave.approximants.precession_msa_reference_l_clean import (
        orbital_angular_momentum_reference_3pn,
    )
    from torchwave.approximants.precession_msa_waveform_clean import (
        transport_msa_angles_for_phi_ref,
    )
    from torchwave.approximants.precession_source_frame_clean import source_frame_geometry

    m1, m2, f_ref = 36.0, 22.0, 40.0
    chi1, chi2 = (0.2, -0.1, 0.35), (-0.08, 0.06, -0.25)
    inclination = 0.65
    phase = float(phi_ref)
    l_ref = float(orbital_angular_momentum_reference_3pn(
        m1, m2, f_ref, chi1[2], chi2[2],
    ))
    n1, n2 = m1/(m1+m2), m2/(m1+m2)
    spin = tuple(n1*n1*a+n2*n2*b for a, b in zip(chi1, chi2))
    initial, final, expected = _scalar_charts(l_ref, spin, inclination, phi_ref)

    def torch_chart(value):
        return source_frame_geometry(
            m1, m2, *chi1, *chi2, inclination, value, f_ref,
            compatibility="v223",
        )

    t0, t1 = torch_chart(0.0), torch_chart(phase)
    j1 = generic_source_frame_geometry_v223(
        m1, m2, jnp.asarray(chi1), jnp.asarray(chi2),
        inclination, phase, f_ref,
    )
    for scalar, torch_value, jax_value in zip(
        (final.alpha0, final.theta_jn, final.zeta, final.phi_j),
        t1,
        (j1[name] for name in ("alpha0", "theta_jn", "zeta", "phi_j")),
    ):
        np.testing.assert_allclose(float(torch_value), float(scalar), rtol=0, atol=3e-14)
        np.testing.assert_allclose(float(jax_value), float(scalar), rtol=0, atol=3e-14)
    np.testing.assert_allclose(float(j1["l_reference"]), l_ref, rtol=0, atol=3e-14)

    angles = {"alpha": torch.tensor([0.1, 0.2], dtype=torch.float64),
              "gamma": torch.tensor([0.0, -0.1], dtype=torch.float64),
              "cos_beta": torch.tensor(0.9, dtype=torch.float64)}
    _, anchor, details = transport_msa_angles_for_phi_ref(
        angles, t0, t1, convention=1,
    )
    np.testing.assert_allclose(float(details["delta_alpha"]), float(expected.delta_alpha),
                               rtol=0, atol=3e-14)
    np.testing.assert_allclose(float(details["delta_gamma"]), float(expected.delta_gamma),
                               rtol=0, atol=3e-14)
    np.testing.assert_allclose(float(anchor["theta_jn"]), float(expected.theta_jn),
                               rtol=0, atol=3e-14)
    np.testing.assert_allclose(float(anchor["zeta"]), float(expected.zeta),
                               rtol=0, atol=3e-14)
    # J is source-fixed while only the observer azimuth changes.  Thus the
    # regular-chart gamma offset vanishes; exact aligned J has no such chart.
    assert abs(expected.delta_gamma) < mpmath.mpf("1e-80")
    assert initial.phi_j_defined and final.phi_j_defined


def test_scalar_transport_exposes_aligned_chart_singularity():
    initial, final = (
        source_chart(("0", "0", ".1"), orbital_l=".8", inclination=".65",
                     phi_ref=phase)
        for phase in ("0", ".3")
    )
    with pytest.raises(ValueError, match="nonzero transverse J"):
        source_chart_transport(initial, final)


def test_scalar_transport_rejects_changed_intrinsic_angular_momentum():
    initial = source_chart((".08", "-.03", ".1"), orbital_l=".8",
                           inclination=".65", phi_ref="0")
    final = source_chart((".09", "-.03", ".1"), orbital_l=".8",
                         inclination=".65", phi_ref=".3")
    with pytest.raises(ValueError, match="same intrinsic J"):
        source_chart_transport(initial, final)


def test_scalar_chart_is_periodic_in_observer_azimuth():
    spin = (".08", "-.03", ".1")
    _, final, _ = _scalar_charts(".8", spin, ".65", "0.3")
    full_turn = source_chart(spin, orbital_l=".8", inclination=".65",
                             phi_ref=str(mpmath.mpf(".3") + 2*mpmath.pi))
    # Decimal input to the scalar oracle carries the precision of the value
    # supplied, so a double-precision 2pi return is compared at that scale.
    assert abs(mpmath.e**(2j*(full_turn.zeta-final.zeta))-1) < 1e-14
    assert abs(mpmath.e**(1j*(full_turn.alpha0-final.alpha0))-1) < 1e-14
