"""Energy-balance check for the published leading-adiabatic initial data."""

import mpmath as mp
import pytest

from dev.native_rom.reference.hamiltonian import (
    aligned_effective_hamiltonian, body_parameters, energy_map, radial_potentials,
)
from dev.native_rom.reference.initial_conditions import leading_adiabatic_state


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
