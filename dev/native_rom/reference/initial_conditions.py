"""Scalar leading-adiabatic initial state for the published EOB Hamiltonian.

The circular conditions and angular-momentum balance follow Buonanno--Damour,
gr-qc/0001013v2, Eqs. (68), (69), and (77) in the arXiv HTML rendering.
The linear radial-velocity relation follows its Eq. (83), converted to the
tortoise momentum of Damour--Nagar, 0704.3550v2, Eqs. (7)--(11).
The supplied torque is an explicit input: this does not validate the native
35-mode flux or reproduce SEOBNRv4's historical initial-data procedure.
"""

import mpmath as mp

from .hamiltonian import aligned_effective_hamiltonian, body_parameters, energy_map, radial_potentials


def leading_adiabatic_state(m1, m2, chi1, chi2, omega, torque, *, digits=50):
    """Return (r, p_r*, p_phi) from a supplied circular torque.

    All quantities use initial-total-mass units; the torque is dp_phi/d(t/M)
    with angular momentum divided by reduced mass times total mass.
    """
    with mp.workdps(digits):
        m1, m2, chi1, chi2, omega, torque = map(
            lambda x: mp.mpf(str(x)), (m1, m2, chi1, chi2, omega, torque)
        )
        if min(m1, m2, omega) <= 0 or abs(chi1) > 1 or abs(chi2) > 1 or torque >= 0:
            raise ValueError("Require positive masses/frequency, physical spins, and inward torque")
        n, a, k = body_parameters(m1, m2, chi1, chi2)

        def h(r, pr, j):
            # mpmath raises working precision during numerical differentiation.
            return aligned_effective_hamiltonian(
                m1, m2, chi1, chi2, r, pr, j, digits=mp.mp.dps
            )

        def circular_residuals(r, j):
            he = h(r, 0, j)
            total, _ = energy_map(n, he, digits=mp.mp.dps)
            return (mp.diff(lambda radius: h(radius, 0, j), r),
                    mp.diff(lambda momentum: h(r, 0, momentum), j)/total - omega)

        guess_r = omega**(-mp.mpf(2)/3)
        guess_j = mp.sqrt(guess_r/(1 - 3/guess_r))
        r, j = mp.findroot(circular_residuals, (guess_r, guess_j), tol=mp.power(10, -digits+8))
        h_rr = mp.diff(lambda radius: h(radius, 0, j), r, 2)
        h_rj = mp.diff(lambda radius, momentum: h(radius, 0, momentum),
                       (r, j), (1, 1))
        h_pp = mp.diff(lambda pr: h(r, pr, j), 0, 2)
        j_prime = -h_rr/h_rj
        if j_prime <= 0:
            raise ValueError("The circular orbit is not on the stable branch")
        total, _ = energy_map(n, h(r, 0, j), digits=mp.mp.dps)
        xi = radial_potentials(r, n, a, k, digits=mp.mp.dps)["xi"]
        pr_star = torque*total/(j_prime*xi*h_pp)
        return +r, +pr_star, +j
