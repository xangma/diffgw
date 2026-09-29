"""Conditional scalar NSBH window, phase, and quadrupole projection."""

from dataclasses import dataclass

import mpmath

from .nsbh_amplitude_window_sum import amplitude_window_sum


@dataclass(frozen=True)
class Polarizations:
    plus: mpmath.mpc
    cross: mpmath.mpc


def compose(frequency_hz, mass_sec, centers_hz, width_mf, carriers,
            phase_rad, strain_scale, inclination_rad, *, dps=70):
    """Compose conditioned Eq. (4) carriers with a supplied phase and scale.

    The overall minus sign fixes the NSBH plus-polarization convention.
    Carrier, phase, joining, and scale provenance are separate gates.
    """
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    phase, scale, inclination = (
        ctx.mpf(value) if isinstance(value, mpmath.mpf) else ctx.mpf(str(value))
        for value in (phase_rad, strain_scale, inclination_rad)
    )
    if not all(ctx.isfinite(value) for value in (phase, scale, inclination)):
        raise ValueError("Expected finite phase, scale and inclination")
    if scale <= 0 or not 0 <= inclination <= ctx.pi:
        raise ValueError("Expected positive scale and inclination in [0, pi]")
    amplitude = scale * amplitude_window_sum(
        frequency_hz, mass_sec, centers_hz, width_mf, carriers, dps=dps,
    ).value
    carrier = amplitude * ctx.exp(-ctx.j * phase)
    cosine = ctx.cos(inclination)
    return Polarizations(-(1 + cosine**2) * carrier / 2, ctx.j * cosine * carrier)
