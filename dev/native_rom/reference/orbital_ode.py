"""Paper-derived orbital vector field at supplied zero-radial-momentum states.

Bohé et al. 1611.03703v3, Sec. II.2 evolves the EOB Hamiltonian with
radiation reaction. Buonanno--Damour gr-qc/0001013v2, Eqs. (24)--(27),
give the polar Hamilton equations. Pan et al. 0912.3466v2, Eq. (28),
supplies the tortoise factor. This scalar check takes the angular torque as
input and sets the radial dissipative force to zero; it is not an initial-data
prescription or a time integrator.
"""

import mpmath as mp

from .hamiltonian import (
    aligned_effective_hamiltonian, body_parameters, energy_map,
    radial_potentials,
)


def _five_point_derivative(function, point):
    """Differentiate the scalar paper Hamiltonian without frontend AD."""
    step = mp.mpf("1e-9") * max(1, abs(point))
    return (-function(point + 2*step) + 8*function(point + step)
            - 8*function(point - step) + function(point - 2*step))/(12*step)


def zero_radial_vector_field(m1, m2, chi1, chi2, r, phi, p_phi, torque,
                             *, digits=60):
    """Return (r-dot, phi-dot, p_r*-dot, p_phi-dot) at supplied p_r*=0.

    The aligned Hamiltonian is even in p_r*, so r-dot vanishes on this
    chart. The supplied torque is the entire angular dissipative force;
    no flux modes, root-finding, or event/step choices enter this leaf.
    """
    with mp.workdps(digits + 20):
        r, phi, p_phi, torque = map(
            lambda value: mp.mpf(str(value)), (r, phi, p_phi, torque))
        if not all(map(mp.isfinite, (r, phi, p_phi, torque))):
            raise ValueError("finite orbital state and torque required")
        n, a, k = body_parameters(m1, m2, chi1, chi2)
        xi = radial_potentials(r, n, a, k, digits=digits + 20)["xi"]

        def h_eff(radius, angular_momentum):
            return aligned_effective_hamiltonian(
                m1, m2, chi1, chi2, radius, 0, angular_momentum,
                digits=digits + 20)

        total, _ = energy_map(n, h_eff(r, p_phi), digits=digits + 20)
        d_h_dr = _five_point_derivative(lambda radius: h_eff(radius, p_phi), r)
        d_h_dj = _five_point_derivative(lambda momentum: h_eff(r, momentum), p_phi)
        return mp.mpf(0), +(d_h_dj/total), +(-xi*d_h_dr/total), +torque
