"""Independent scalar diagnostics for printed NSBH amplitude variants.

The paper and compatibility variants are intentionally separate.  Agreement
with the latter does not qualify the former or the full amplitude.
"""

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class VariantPair:
    paper: mpmath.mpf
    compatibility: mpmath.mpf

    @property
    def project_correction(self) -> mpmath.mpf:
        """Alias for the smooth polynomial continuation / project_correction variant."""
        return self.compatibility


def _context(values, dps):
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    converted = tuple(ctx.mpf(value) if isinstance(value, mpmath.mpf) else ctx.mpf(str(value)) for value in values)
    if not all(ctx.isfinite(value) for value in converted):
        raise ValueError("Expected finite NSBH variant inputs")
    return ctx, converted


def low_no_torus_pn_center(q, f_rd_tilde, f_tide, epsilon_ins, *, dps=70):
    """Thompson Table I / Pannarale Eq. 26 versus the outer-factor adapter."""
    _, (q, fr, ft, eps) = _context((q, f_rd_tilde, f_tide, epsilon_ins), dps)
    if q < 1 or fr <= 0 or ft <= 0:
        raise ValueError("Expected q>=1 and positive frequencies")
    paper = ((q - 1) * fr + eps * ft) / q
    return VariantPair(paper, eps * paper)


def gamma1_small_lambda(lambda_ns, *, dps=70):
    r"""Thompson Eq. 26 polynomial versus the project_correction polynomial.

    Thompson (2020) Eq. (26) prints:
      \gamma'_1 = 1 - 0.5*\Lambda - 0.25*\Lambda^2   for \Lambda <= 1
      \gamma'_1 = 1.25                               for \Lambda > 1
    This causes an unphysical jump discontinuity of +1.0 at \Lambda = 1 (from 0.25 to 1.25).

    The smooth polynomial continuation variant (labeled 'project_correction' or
    'compatibility') is:
      1 + 0.5*\Lambda - 0.25*\Lambda^2               for \Lambda <= 1
      1.25                                           for \Lambda > 1
    which connects smoothly to \Lambda=0 (BBH limit g(0)=1) and with C^1 continuity
    to \Lambda > 1 (g(1)=1.25, g'(1)=0).
    """
    _, (lam,) = _context((lambda_ns,), dps)
    if not 0 <= lam <= 1:
        raise ValueError("Expected 0<=Lambda<=1")
    return VariantPair(1 - lam / 2 - lam**2 / 4, 1 + lam / 2 - lam**2 / 4)


def delta2_large_lambda(f_tide, f_rd_tilde, *, dps=70):
    r"""Thompson Eq. 29 minus window versus the project_correction plus window.

    Thompson (2020) Eq. (29) prints \omega^- (minus window):
      0.81248 * 0.5 * (1 - \tanh(4*(x - x_3)/d_3))
    which decays to zero as f_tide >> f_rd_tilde (BBH limit).

    The project_correction variant uses +0.81248 \omega^+ (plus window):
      0.81248 * 0.5 * (1 + \tanh(4*(x - x_3)/d_3))
    which preserves the BBH ringdown width constant \delta_2 = 0.81248 as x -> +\infty.
    """
    ctx, (ft, fr) = _context((f_tide, f_rd_tilde), dps)
    if ft <= 0 or fr <= 0:
        raise ValueError("Expected positive frequencies")
    x = (ft - fr) / fr
    argument = 4 * (x - ctx.mpf("0.0188092")) / ctx.mpf("0.338737")
    window = ctx.tanh(argument)
    scale = ctx.mpf("0.81248") / 2
    return VariantPair(scale * (1 - window), scale * (1 + window))
