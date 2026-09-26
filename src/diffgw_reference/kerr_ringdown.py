"""Scalar, high-precision energy-normalized damped-mode projection.

The QNM angular frequency, damping rate, and two angular factors are explicit
inputs.  This reference checks their composition into a waveform, not the
independent correctness of the QNM or spheroidal-harmonic calculations.
Inputs for mass and distance are geometric seconds (GM/c^3 and D/c).
"""

import mpmath as mp


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
