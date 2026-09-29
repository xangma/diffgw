"""Independent scalar reference for a bounded eccentric Keplerian EMRI limit.

Peters & Mathews (1963); Peters (1964) Eqs. (5.6)-(5.7);
Barack & Cutler (2004) Sec. III B, Eqs. (7)-(10), (27).
No native backend or external waveform implementation is imported.
"""

import mpmath as mp
import sympy as sp

from diffgw_reference import emri_peters_flux as flux

SPEC_VERSION = "emri-newtonian-eccentric-quadrupole/v1"
COMPONENT_IDS = {
    "orbit": "emri.orbit.peters_kepler_rk4.v1",
    "harmonics": "emri.harmonics.peters_mathews.v1",
    "observer": "emri.observer.fixed_polarizations.v1",
}
SOURCE = "https://arxiv.org/pdf/gr-qc/0310125"

n = sp.symbols("n", integer=True, positive=True)
e, phi, amp, u, gamma, distance = sp.symbols(
    "e phi A u gamma D", real=True)
j = lambda k: sp.besselj(k, n*e)
an = -n*amp*(j(n-2)-2*e*j(n-1)+sp.Rational(2, 1)/n*j(n)
             +2*e*j(n+1)-j(n+2))*sp.cos(n*phi)
bn = -n*amp*sp.sqrt(1-e**2)*(j(n-2)-2*j(n)+j(n+2))*sp.sin(n*phi)
cn = 2*amp*j(n)*sp.cos(n*phi)
plus_n = (-(1+u**2)*(an*sp.cos(2*gamma)-bn*sp.sin(2*gamma))
          +(1-u**2)*cn)/distance
cross_n = 2*u*(bn*sp.cos(2*gamma)+an*sp.sin(2*gamma))/distance


def circular_limit():
    """Exact n=2 circular result in the chosen polarization convention."""
    return (sp.trigsimp(plus_n.subs({n: 2, e: 0})),
            sp.trigsimp(cross_n.subs({n: 2, e: 0})))


_harmonic = sp.lambdify((n, e, phi, amp, u, gamma, distance),
                       (plus_n, cross_n), "mpmath")
_dp = sp.lambdify((flux.M, flux.MU, flux.P, flux.E),
                 flux.semilatus_rectum_rate_expr(), "mpmath")
_de = sp.lambdify((flux.M, flux.MU, flux.P, flux.E),
                 flux.eccentricity_rate_p_expr(), "mpmath")


def trajectory(times, mass_solar, secondary_solar, p0, e0, phase0=0, dps=70):
    """Independent high-precision RK4 on the supplied positive-increment grid."""
    with mp.workdps(dps):
        ts = [mp.mpf(str(t)) for t in times]
        if len(ts) < 2 or any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("times must be strictly increasing")
        solar_second = mp.mpf(str(flux.MSUN_SEC.evalf(dps)))
        m = mp.mpf(str(mass_solar))*solar_second
        mu = mp.mpf(str(secondary_solar))*solar_second
        y = tuple(map(lambda x: mp.mpf(str(x)), (p0, e0, phase0)))

        def rhs(state):
            p, ecc, _ = state
            return (_dp(m, mu, p, ecc), _de(m, mu, p, ecc),
                    (1-ecc**2)**mp.mpf("1.5")/(m*p**mp.mpf("1.5")))

        samples = [y]
        for start, end in zip(ts, ts[1:]):
            h = end-start
            k1 = rhs(y)
            k2 = rhs(tuple(v+h*k/2 for v, k in zip(y, k1)))
            k3 = rhs(tuple(v+h*k/2 for v, k in zip(y, k2)))
            k4 = rhs(tuple(v+h*k for v, k in zip(y, k3)))
            y = tuple(v+h*(a+2*b+2*c+d)/6
                      for v, a, b, c, d in zip(y, k1, k2, k3, k4))
            samples.append(y)
        return samples


def polarizations(times, mass_solar=1e6, secondary_solar=10, p0=16,
                  e0=0.2, inclination=0.7, periapsis=0, phase0=0,
                  distance_mpc=1000, n_harmonics=5, dps=70):
    """Source-frame (h+,hx), with no LISA or other detector response."""
    with mp.workdps(dps):
        states = trajectory(times, mass_solar, secondary_solar, p0, e0, phase0, dps)
        mu = mp.mpf(str(secondary_solar))*mp.mpf(str(flux.MSUN_SEC.evalf(dps)))
        # SI parsecs and c are defined independently of native constants.
        d = mp.mpf(str(distance_mpc))*mp.mpf("3.0856775814913673e22")/mp.mpf(299792458)
        sight = mp.cos(mp.mpf(str(inclination)))
        gamma_value = mp.mpf(str(periapsis))
        result = []
        for p, ecc, phase in states:
            a = mu*(1-ecc**2)/p
            hp = hx = mp.mpf(0)
            for harmonic in range(1, n_harmonics+1):
                value_plus, value_cross = _harmonic(
                    harmonic, ecc, phase, a, sight, gamma_value, d)
                hp += value_plus
                hx += value_cross
            result.append((+hp, +hx))
        return result
