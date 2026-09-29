"""Independent reference gates for the unresolved generic aligned boundary."""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from diffgw_reference.xp_msa_generic_boundary import generic_cubic_phase, source_chart


def _reference(scale, *, dps=120):
    scale = mpmath.mpf(scale)
    return generic_cubic_phase(
        "36", "22", (str(scale), "0", ".35"),
        (str(-scale / 2), "0", "-.25"), f_ref="40", dps=dps,
    )


def test_published_cubic_resolves_the_near_aligned_scaling_without_a_clamp():
    scales = tuple(mpmath.mpf(value) for value in ("1e-4", "1e-6", "1e-8", "1e-10"))
    diagnostics = tuple(_reference(str(scale)) for scale in scales)
    gap_coefficients = tuple(
        diagnostic.physical_root_gap / scale**2
        for diagnostic, scale in zip(diagnostics, scales)
    )
    endpoint_coefficients = tuple(
        (1 + diagnostic.acos_argument) / scale**4
        for diagnostic, scale in zip(diagnostics, scales)
    )

    # Derived for this fixture: the physical root gap is O(chi_perp^2), but
    # the acos endpoint distance carrying that split is O(chi_perp^4).
    assert abs(gap_coefficients[-1] - mpmath.mpf("0.6894485975914909")) < mpmath.mpf("2e-16")
    assert abs(endpoint_coefficients[-1] - mpmath.mpf("3296.9252383264067")) < mpmath.mpf("5e-13")
    for coefficient in gap_coefficients:
        assert abs(coefficient / gap_coefficients[-1] - 1) < mpmath.mpf("4e-7")
    for coefficient in endpoint_coefficients:
        assert abs(coefficient / endpoint_coefficients[-1] - 1) < mpmath.mpf("2e-6")

    for diagnostic in diagnostics:
        assert diagnostic.physical_root_gap > 0
        assert abs(diagnostic.phase_fraction - 1) < mpmath.mpf("1e-80")
        with mpmath.workdps(120):
            assert mpmath.almosteq(
                diagnostic.initial_phase,
                mpmath.ellipk(diagnostic.elliptic_parameter),
                rel_eps=mpmath.mpf("1e-100"),
            )
        assert max(abs(value) for value in diagnostic.root_residuals) < mpmath.mpf("1e-115")


def test_appendix_c_chart_keeps_the_exact_aligned_azimuth_undefined():
    exact = source_chart(
        ("0", "0", ".1"), orbital_l=".8", inclination=".65", phi_ref=".3",
    )
    near = source_chart(
        ("1e-10", "0", ".1"), orbital_l=".8", inclination=".65", phi_ref=".3",
    )
    assert not exact.phi_j_defined
    assert exact.phi_j is None
    assert near.phi_j_defined
    assert near.phi_j == 0
    assert abs(near.alpha0 - (mpmath.pi / 2 + mpmath.mpf(".3"))) < mpmath.mpf("2e-10")
    assert abs(near.theta_jn - mpmath.mpf(".65")) < mpmath.mpf("4e-11")
    # +pi and -pi are the same spin-2 polarization basis.
    assert abs(mpmath.e**(2j * near.zeta) - mpmath.e**(2j * exact.zeta)) < mpmath.mpf("4e-10")


def test_torch_and_jax_diagnostics_match_where_float64_resolves_the_cubic():
    pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from jaxwave.approximants.precession_msa_v223_generic_boundary import (
        generic_cubic_boundary_diagnostic as jax_diagnostic,
    )
    from torchwave.approximants.precession_msa_v223_generic_boundary import (
        generic_cubic_boundary_diagnostic as torch_diagnostic,
    )

    spin1, spin2 = (0.02, 0.0, 0.35), (-0.01, 0.0, -0.25)
    torch_values = torch_diagnostic(36.0, 22.0, spin1, spin2, 40.0)
    jax_values = jax_diagnostic(36.0, 22.0, spin1, spin2, 40.0)
    for name in (
        "coefficients", "depressed_coefficients", "acos_argument",
        "acos_argument_used", "compatibility_roots", "compatibility_root_gap",
        "total_spin_squared", "phase_fraction_unclipped",
    ):
        np.testing.assert_allclose(
            np.asarray(jax_values[name]), torch_values[name].detach().numpy(),
            rtol=3e-13, atol=3e-16, err_msg=name,
        )

    scalar = _reference(".02")
    np.testing.assert_allclose(
        torch_values["coefficients"].detach().numpy(),
        np.asarray(tuple(float(value) for value in scalar.coefficients)),
        rtol=3e-14, atol=3e-16,
    )
    assert math.isclose(
        torch_values["compatibility_root_gap"].item(),
        float(scalar.physical_root_gap), rel_tol=1e-10, abs_tol=0.0,
    )


def test_float64_diagnostics_expose_instead_of_hide_the_unresolved_root_split():
    pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from jaxwave.approximants.precession_msa_v223_generic_boundary import (
        generic_cubic_boundary_diagnostic as jax_diagnostic,
    )
    from torchwave.approximants.precession_msa_v223_generic_boundary import (
        generic_cubic_boundary_diagnostic as torch_diagnostic,
    )

    spin1, spin2 = (1e-10, 0.0, 0.35), (-5e-11, 0.0, -0.25)
    scalar = _reference("1e-10")
    torch_values = torch_diagnostic(36.0, 22.0, spin1, spin2, 40.0)
    jax_values = jax_diagnostic(36.0, 22.0, spin1, spin2, 40.0)
    scalar_endpoint_distance = float(1 + scalar.acos_argument)
    scalar_gap = float(scalar.physical_root_gap)

    for values, convert in (
        (torch_values, lambda value: value.detach().numpy()),
        (jax_values, np.asarray),
    ):
        raw = float(convert(values["acos_argument"]))
        gap = abs(float(convert(values["compatibility_root_gap"])))
        # The exact endpoint distance is ~3.30e-37 and the physical root gap
        # ~6.89e-21.  A float64 continuation cannot represent both here.
        assert abs((1 + raw) - scalar_endpoint_distance) > 1e10 * scalar_endpoint_distance
        assert gap == 0.0 or max(gap / scalar_gap, scalar_gap / gap) > 1e3
        used = float(convert(values["acos_argument_used"]))
        assert -1.0 <= used <= 1.0
        if raw != used:
            assert abs(raw) > 1.0
