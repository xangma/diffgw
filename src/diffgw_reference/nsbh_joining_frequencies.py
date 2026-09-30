"""Scalar transcription of Thompson et al. (2020), Table I frequency joins."""

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class JoiningFrequencies:
    f0: mpmath.mpf
    f1: mpmath.mpf
    f2: mpmath.mpf | None


def joining_frequencies(q, f_tide, f_rd, torus_ratio, f_rd_tilde, epsilon_ins, *, dps=70):
    """Evaluate the paper's four rows; dashes for low-branch f2 become None."""
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    q, ft, fr, t, fr_tilde, eps = (
        ctx.mpf(value) if isinstance(value, mpmath.mpf) else ctx.mpf(str(value))
        for value in (q, f_tide, f_rd, torus_ratio, f_rd_tilde, epsilon_ins)
    )
    if not all(ctx.isfinite(value) for value in (q, ft, fr, t, fr_tilde, eps)):
        raise ValueError("Expected finite joining-frequency inputs")
    if q < 1 or ft <= 0 or fr <= 0 or fr_tilde <= 0 or t < 0:
        raise ValueError("Expected q>=1, positive frequencies, and torus>=0")

    if ft >= fr:
        center = eps * fr_tilde if t > 0 else fr_tilde
        return JoiningFrequencies(center, center, fr_tilde)
    if t > 0:
        return JoiningFrequencies(eps * ft, ft, None)
    return JoiningFrequencies(((q - 1) * fr_tilde + eps * ft) / q, ((q - 1) * fr_tilde + ft) / q, None)
