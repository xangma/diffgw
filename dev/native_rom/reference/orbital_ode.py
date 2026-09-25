"""Paper-derived orbital vector field at supplied aligned-spin states.

Bohé et al. 1611.03703v3, Sec. II.2 evolves the EOB Hamiltonian with
radiation reaction. Buonanno--Damour gr-qc/0001013v2, Eqs. (24)--(27),
give the polar Hamilton equations. Pan et al. 0912.3466v2, Eq. (28),
supplies the tortoise factor. This scalar module checks both supplied-force
and paper-derived pre-NQC flux variants. It is not an initial-data
prescription or a time integrator.
"""

import mpmath as mp

from .hamiltonian import (
    aligned_effective_hamiltonian, body_parameters, energy_map,
    radial_potentials,
)
from .flux import emitted_flux, factorized_mode_amplitudes


def _five_point_derivative(function, point):
    """Differentiate the scalar paper Hamiltonian without frontend AD."""
    step = mp.mpf("1e-9") * max(1, abs(point))
    return (-function(point + 2*step) + 8*function(point + step)
            - 8*function(point - step) + function(point - 2*step))/(12*step)


def supplied_force_vector_field(m1, m2, chi1, chi2, r, phi, p_r_star,
                                p_phi, radial_force, torque, *, digits=60):
    """Return (r-dot, phi-dot, p_r*-dot, p_phi-dot) at a supplied state.

    The two dissipative forces are caller inputs. Differentiation holds the
    other state coordinates fixed, including p_r* for the radial partial.
    No flux modes, root-finding, or event/step choices enter this leaf.
    """
    with mp.workdps(digits + 20):
        r, phi, p_r_star, p_phi, radial_force, torque = map(
            lambda value: mp.mpf(str(value)),
            (r, phi, p_r_star, p_phi, radial_force, torque))
        if not all(map(mp.isfinite, (r, phi, p_r_star, p_phi,
                                    radial_force, torque))):
            raise ValueError("finite orbital state and forces required")
        n, a, k = body_parameters(m1, m2, chi1, chi2)
        xi = radial_potentials(r, n, a, k, digits=digits + 20)["xi"]

        def h_eff(radius, radial_momentum, angular_momentum):
            return aligned_effective_hamiltonian(
                m1, m2, chi1, chi2, radius, radial_momentum,
                angular_momentum,
                digits=digits + 20)

        total, _ = energy_map(n, h_eff(r, p_r_star, p_phi), digits=digits + 20)
        d_h_dr = _five_point_derivative(
            lambda radius: h_eff(radius, p_r_star, p_phi), r)
        d_h_dp = _five_point_derivative(
            lambda momentum: h_eff(r, momentum, p_phi), p_r_star)
        d_h_dj = _five_point_derivative(
            lambda momentum: h_eff(r, p_r_star, momentum), p_phi)
        return (+xi*d_h_dp/total, +d_h_dj/total,
                -xi*d_h_dr/total + radial_force, +torque)


def zero_radial_vector_field(m1, m2, chi1, chi2, r, phi, p_phi, torque,
                             *, digits=60):
    """Return the supplied-force vector field at p_r*=0 and zero radial force."""
    return supplied_force_vector_field(
        m1, m2, chi1, chi2, r, phi, 0, p_phi, 0, torque, digits=digits)


def paper_flux_vector_field(m1, m2, chi1, chi2, r, phi, p_r_star, p_phi,
                            *, spin_truncation, digits=60):
    """Compose the independent scalar Hamiltonian and all pre-NQC modes.

    The angular force follows Bohé Eq. (9) in reduced-mass units. Its radial
    component follows the specified p_r*/p_phi radiation-reaction gauge.
    Mode spin/truncation choices are conditional, not historical-v4 claims.
    """
    with mp.workdps(digits):
        r, phi, p_r_star, p_phi = map(
            lambda value: mp.mpf(str(value)), (r, phi, p_r_star, p_phi))
        if p_phi <= 0:
            raise ValueError("positive p_phi is required")
        nu, _, _ = body_parameters(m1, m2, chi1, chi2)
        omega = supplied_force_vector_field(
            m1, m2, chi1, chi2, r, phi, p_r_star, p_phi, 0, 0,
            digits=digits)[1]
        omega_zero_radial = supplied_force_vector_field(
            m1, m2, chi1, chi2, r, phi, 0, p_phi, 0, 0,
            digits=digits)[1]
        heff = aligned_effective_hamiltonian(
            m1, m2, chi1, chi2, r, p_r_star, p_phi, digits=digits)
        hreal = energy_map(nu, heff, digits=digits)[0]
        modes = factorized_mode_amplitudes(
            m1, m2, chi1, chi2, omega, omega_zero_radial,
            p_phi, heff, hreal, spin_truncation=spin_truncation, digits=digits)
        flux = emitted_flux(omega, modes, digits=digits)
        torque = -flux/(nu*omega)
        radial_force = p_r_star/p_phi*torque
        return supplied_force_vector_field(
            m1, m2, chi1, chi2, r, phi, p_r_star, p_phi,
            radial_force, torque, digits=digits)
