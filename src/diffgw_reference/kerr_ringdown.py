"""Scalar, high-precision energy-normalized damped-mode projection.

The QNM angular frequency, damping rate, and two angular factors are explicit
inputs.  This reference checks their composition into a waveform, not the
independent correctness of the QNM or spheroidal-harmonic calculations.
Inputs for mass and distance are geometric seconds (GM/c^3 and D/c).
"""

import mpmath as mp

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from .kerr_angular import spheroidal_projection
from .kerr_qnm import ScalarQNM220


SPEC_VERSION = "kerr-ringdown-220-energy-direct-class/v1"
COMPONENT_IDS = {
    name: f"kerr.direct220.{name}.v1"
    for name in ("qnm", "angular", "energy_amplitude", "causal_td_fd")
}


def _real(value):
    return mp.mpf(str(value))


def _complex(value):
    return mp.mpc(str(value.real), str(value.imag))


def energy_amplitude(mass_seconds, distance_seconds, energy_fraction, omega, damping_rate):
    """Single complex-mode coefficient for the finite-Q energy integral."""
    mass, distance, energy = map(_real, (mass_seconds, distance_seconds, energy_fraction))
    frequency, decay = map(_real, (omega, damping_rate))
    return mp.sqrt(32 * mp.pi * energy * mass / decay) / (
        distance * mp.sqrt(1 + (frequency / decay) ** 2)
    )


def time_polarizations(time, onset, amplitude, omega, damping_rate, angular_plus, angular_mirror):
    """Real polarizations of a causal mode and its equatorial mirror."""
    u = _real(time) - _real(onset)
    if u < 0:
        return mp.mpf(0), mp.mpf(0)
    carrier = mp.exp(-(_real(damping_rate) + 1j * _real(omega)) * u)
    pos, mirror = _complex(angular_plus), _complex(angular_mirror)
    scale = _real(amplitude) / mp.sqrt(2)
    return -scale * mp.re(carrier * (pos + mirror)), scale * mp.im(carrier * (pos - mirror))


def frequency_polarizations(frequency, onset, amplitude, omega, damping_rate,
                            angular_plus, angular_mirror):
    """Fourier-minus transform obtained from the mode's Laplace integral."""
    w = 2 * mp.pi * _real(frequency)
    decay, oscillation = _real(damping_rate), _real(omega)
    pos, mirror = _complex(angular_plus), _complex(angular_mirror)
    sum_shape, difference_shape = pos + mirror, pos - mirror
    positive_pole = 1 / (decay + 1j * (w + oscillation))
    negative_pole = 1 / (decay + 1j * (w - oscillation))
    shift = mp.exp(-1j * w * _real(onset))
    scale = _real(amplitude) / (2 * mp.sqrt(2))
    return (
        -scale * shift * (sum_shape * positive_pole + mp.conj(sum_shape) * negative_pole),
        scale * shift * (difference_shape * positive_pole - mp.conj(difference_shape) * negative_pole) / 1j,
    )


def frequency_by_quadrature(frequency, onset, amplitude, omega, damping_rate,
                            angular_plus, angular_mirror):
    """Numerically integrate the causal time polarizations; a second FD route."""
    decay = _real(damping_rate)
    w = 2 * mp.pi * _real(frequency)
    shift = mp.exp(-1j * w * _real(onset))

    def integral(index):
        def integrand(u):
            polarizations = time_polarizations(
                u, 0, amplitude, omega, damping_rate, angular_plus, angular_mirror
            )
            return polarizations[index] * mp.exp(-1j * w * u)

        intervals = [2 * k / decay for k in range(21)]
        return shift * mp.quad(integrand, intervals)

    return integral(0), integral(1)


def direct_class_220_inputs(final_mass, final_spin, distance, energy_fraction,
                            inclination, phi, *, ell_max=10, order=32, depth=400):
    """Independently solve the QNM and angular projection for the direct class.

    Mass is in solar masses, distance in Mpc, angles in radians. The returned
    amplitude is the native *pair* coefficient, not the single-mode coefficient
    used by the conditional public-clean reference above.
    """
    spin = _real(final_spin)
    mass = _real(final_mass) * _real(MSUN_SEC)
    qnm, _ = ScalarQNM220(ell_max=ell_max, order=order, dps=mp.mp.dps).root(
        spin, depth=depth
    )
    omega, decay = mp.re(qnm) / mass, -mp.im(qnm) / mass
    d_plus, d_cross = spheroidal_projection(
        2, 2, spin * qnm, _real(inclination), _real(phi),
        ell_max=ell_max, order=order,
    )
    single = energy_amplitude(
        mass, _real(distance) * _real(MPC_SI) / _real(C_SI),
        energy_fraction, omega, decay,
    )
    return mp.sqrt(2) * single, omega, decay, d_plus, d_cross


def direct_class_220_time(time, onset, amplitude, omega, damping_rate,
                          d_plus, d_cross):
    """Causal direct-class 220 polarizations from independently supplied factors."""
    u = _real(time) - _real(onset)
    if u < 0:
        return mp.mpf(0), mp.mpf(0)
    carrier = _real(amplitude) * mp.exp((-_real(damping_rate) + 1j * _real(omega)) * u)
    return mp.re(carrier * d_plus), mp.re(carrier * d_cross)


def direct_class_220_frequency(frequency, onset, amplitude, omega, damping_rate,
                               d_plus, d_cross):
    """Fourier-minus transform of the direct-class causal mode and mirror."""
    w = 2 * mp.pi * _real(frequency)
    decay, oscillation = _real(damping_rate), _real(omega)
    shift = mp.exp(-1j * w * _real(onset))

    def one(projection):
        return _real(amplitude) * shift / 2 * (
            projection / (decay + 1j * (w - oscillation))
            + mp.conj(projection) / (decay + 1j * (w + oscillation))
        )

    return one(d_plus), one(d_cross)
