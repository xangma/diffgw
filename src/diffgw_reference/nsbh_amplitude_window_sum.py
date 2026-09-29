"""Independent scalar Thompson et al. (2020) NSBH amplitude window sum."""

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class WindowSum:
    value: mpmath.mpf
    frequency_derivative: mpmath.mpf
    carrier_partials: tuple[mpmath.mpf, mpmath.mpf, mpmath.mpf]


def amplitude_window_sum(frequency_hz, mass_sec, centers_hz, width_mf, carriers, *, dps=70):
    """Evaluate Thompson Eq. (4)/(30) with supplied amplitude carriers.

    The frequency derivative holds the three supplied carriers fixed.
    """
    if len(centers_hz) != 3 or len(carriers) != 3:
        raise ValueError("Expected three centers and three carriers")
    ctx = mpmath.mp.clone()
    ctx.dps = dps

    def scalar(value):
        return ctx.mpf(value) if isinstance(value, mpmath.mpf) else ctx.mpf(str(value))

    f, mass, width = (scalar(value) for value in (frequency_hz, mass_sec, width_mf))
    centers = tuple(scalar(value) for value in centers_hz)
    amplitudes = tuple(scalar(value) for value in carriers)
    if not all(ctx.isfinite(value) for value in (f, mass, width, *centers, *amplitudes)):
        raise ValueError("Expected finite amplitude-window inputs")
    if f <= 0 or mass <= 0 or width <= 0 or any(center <= 0 for center in centers):
        raise ValueError("Expected positive frequency, centers, mass and width")
    if any(amplitude < 0 for amplitude in amplitudes):
        raise ValueError("Expected nonnegative supplied carrier amplitudes")

    z = tuple(4 * mass * (f - center) / width for center in centers)
    windows = ((1 - ctx.tanh(z[0])) / 2, (1 - ctx.tanh(z[1])) / 2, (1 + ctx.tanh(z[2])) / 2)
    value = sum(amplitude * window for amplitude, window in zip(amplitudes, windows, strict=True))
    derivative = 2 * mass / width * (
        -amplitudes[0] / ctx.cosh(z[0]) ** 2
        - amplitudes[1] / ctx.cosh(z[1]) ** 2
        + amplitudes[2] / ctx.cosh(z[2]) ** 2
    )
    return WindowSum(value, derivative, windows)
