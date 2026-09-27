"""Energy-balance check for the published leading-adiabatic initial data."""

import json
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import pytest

from dev.native_rom.reference.hamiltonian import (
    aligned_effective_hamiltonian, body_parameters, energy_map, radial_potentials,
)
from dev.native_rom.reference.initial_conditions import (
    leading_adiabatic_state, paper_flux_leading_adiabatic_state,
)


_DOMAIN_PLAN = json.loads(
    (
        Path(__file__).parents[2]
        / "evidence/2026-09-27/ic_paper_flux_domain_plan.json"
    ).read_text()
)
_DOMAIN_ROWS = [
    pytest.param(
        row["m1"],
        row["m2"],
        row["chi1"],
        row["chi2"],
        row["omega"],
        id=row["name"],
    )
    for row in _DOMAIN_PLAN["rows"]
]


@lru_cache
def _paper_flux_state(m1, m2, chi1, chi2, omega, digits):
    configuration = _DOMAIN_PLAN["configuration"]
    return paper_flux_leading_adiabatic_state(
        m1,
        m2,
        chi1,
        chi2,
        omega,
        spin_truncation=configuration["spin_truncation"],
        mode_21_higher_spin=configuration["mode_21_higher_spin"],
        mode_33_higher_spin=configuration["mode_33_higher_spin"],
        digits=digits,
    )


@pytest.mark.parametrize("m1,m2,chi1,chi2,omega", [
    (30, 20, 0, 0, "0.04"),
    (45, 5, "-0.8", "0.3", "0.025"),
])
def test_leading_adiabatic_energy_balance_against_circular_sequence(
    m1, m2, chi1, chi2, omega,
):
    """Check BCD (2006) Eqs. (70)--(77) without reusing the analytic j' formula.

    Neighboring circular angular momenta are solved afresh. Their energy
    slope and the finite-radial-momentum Hamilton velocity give an independent
    energy-balance check of the initialized tortoise momentum.
    """
    with mp.workdps(45):
        torque = mp.mpf("-1e-6")
        r, pr_star, j = leading_adiabatic_state(
            m1, m2, chi1, chi2, omega, torque, digits=45,
        )
        nu, chi_kerr, k = body_parameters(m1, m2, chi1, chi2)

        def h(radius, momentum, angular_momentum):
            return aligned_effective_hamiltonian(
                m1, m2, chi1, chi2, radius, momentum, angular_momentum,
                digits=mp.mp.dps,
            )

        def circular_energy(radius):
            angular_momentum = mp.findroot(
                lambda value: mp.diff(lambda x: h(x, 0, value), radius),
                j, tol=mp.mpf("1e-35"),
            )
            return energy_map(nu, h(radius, 0, angular_momentum), digits=mp.mp.dps)[1]

        def centered_energy_slope(step):
            return (circular_energy(r + step) - circular_energy(r - step)) / (2*step)

        step = mp.mpf("0.002")
        slope = (4*centered_energy_slope(step/2) - centered_energy_slope(step))/3
        total = energy_map(nu, h(r, pr_star, j), digits=mp.mp.dps)[0]
        xi = radial_potentials(r, nu, chi_kerr, k, digits=mp.mp.dps)["xi"]
        radial_velocity = xi*mp.diff(lambda momentum: h(r, momentum, j), pr_star)/total
        emitted_energy = mp.mpf(omega)*torque

        assert radial_velocity < 0
        assert slope > 0
        assert abs((radial_velocity*slope - emitted_energy)/emitted_energy) < mp.mpf("2e-9")


@pytest.mark.parametrize("m1,m2,chi1,chi2,omega", _DOMAIN_ROWS)
@pytest.mark.parametrize("frontend", ["torch", "jax"])
def test_paper_flux_initial_state_against_frontend(
    frontend, m1, m2, chi1, chi2, omega,
):
    """Check the frozen domain grid against the scalar 35-mode composition."""
    expected = _paper_flux_state(
        m1, m2, chi1, chi2, omega,
        _DOMAIN_PLAN["configuration"]["reference_digits"],
    )
    r, pr_star, p_phi, emitted, torque = map(float, expected)
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    assert emitted > 0 and torque < 0 and pr_star < 0
    gates = _DOMAIN_PLAN["gates"]
    assert torque == pytest.approx(
        -emitted/(nu*omega), rel=gates["torque_balance_relative_error_at_most"]
    )

    if frontend == "torch":
        import torch
        from torchwave.approximants.eob.initial_conditions import (
            find_quasicircular_initial_conditions,
        )
        from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
        from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

        scalar = lambda value: torch.tensor(value, dtype=torch.float64)
        ham = SEOBNRv4Hamiltonian(scalar(nu), scalar(chi1), scalar(chi2),
                                  delta=scalar(delta))
        flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated")
        actual = find_quasicircular_initial_conditions(ham, flux, scalar(omega))
        _, actual_torque = flux.radiation_reaction_forces(
            scalar(omega), scalar(0), actual[2], r=actual[0], hamiltonian=ham)
    else:
        import jax
        import jax.numpy as jnp
        from jaxwave.approximants.eob.initial_conditions import (
            find_quasicircular_initial_conditions,
        )
        from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
        from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

        jax.config.update("jax_enable_x64", True)
        scalar = lambda value: jnp.asarray(value, dtype=jnp.float64)
        ham = SEOBNRv4Hamiltonian(scalar(nu), scalar(chi1), scalar(chi2),
                                  delta=scalar(delta))
        flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated")
        actual = find_quasicircular_initial_conditions(ham, flux, scalar(omega))
        _, actual_torque = flux.radiation_reaction_forces(
            scalar(omega), scalar(0), actual[2], r=actual[0], hamiltonian=ham)

    for value, reference in zip(actual, (r, pr_star, p_phi)):
        assert float(value) == pytest.approx(
            reference,
            rel=gates["frontend_component_relative_error_at_most"],
            abs=gates["frontend_component_absolute_error_at_most"],
        )
    assert float(actual_torque) == pytest.approx(
        torque,
        rel=gates["frontend_component_relative_error_at_most"],
        abs=gates["frontend_component_absolute_error_at_most"],
    )


@pytest.mark.parametrize("m1,m2,chi1,chi2,omega", _DOMAIN_ROWS)
def test_paper_flux_initial_state_scalar_precision_stability(
    m1, m2, chi1, chi2, omega,
):
    """Require the frozen scalar reference to be stable under +20 digits."""
    configuration = _DOMAIN_PLAN["configuration"]
    lower = _paper_flux_state(
        m1, m2, chi1, chi2, omega, configuration["reference_digits"]
    )
    higher = _paper_flux_state(
        m1, m2, chi1, chi2, omega, configuration["precision_check_digits"]
    )
    relative_errors = [
        abs(value - reference) / max(abs(reference), mp.mpf("1e-100"))
        for value, reference in zip(lower, higher)
    ]
    assert max(relative_errors) <= mp.mpf(
        str(_DOMAIN_PLAN["gates"]["scalar_50_to_70_digit_relative_error_at_most"])
    )
