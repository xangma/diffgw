"""Scalar Thompson et al. (2020) branch-amplitude factors from printed equations."""

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class BranchAmplitudeFits:
    f_rd_tilde: mpmath.mpf
    x_nd: mpmath.mpf
    x_prime_nd: mpmath.mpf
    x_d: mpmath.mpf
    x_prime_d: mpmath.mpf
    epsilon_tide: mpmath.mpf
    epsilon_ins: mpmath.mpf
    sigma_tide: mpmath.mpf


def branch_amplitude_fits(q, chi, compactness, f_tide, f_rd, torus_ratio, lambda_ns, *, dps=70):
    """Evaluate Appendix A Eqs. (20)--(24), (30) and Table I independently."""
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    q, chi, c, ft, fr, t, lam = (
        ctx.mpf(value) if isinstance(value, mpmath.mpf) else ctx.mpf(str(value))
        for value in (q, chi, compactness, f_tide, f_rd, torus_ratio, lambda_ns)
    )
    if not all(ctx.isfinite(value) for value in (q, chi, c, ft, fr, t, lam)):
        raise ValueError("Expected finite branch-amplitude inputs")
    if q < 1 or c <= 0 or c >= ctx.mpf("0.5") or ft <= 0 or fr <= 0 or t < 0 or lam < 0:
        raise ValueError("Expected q>=1, 0<C<1/2, positive frequencies, torus>=0, Lambda>=0")

    eta = q / (1 + q) ** 2
    fr_tilde = (
        ctx.mpf("0.98") * fr * (ctx.mpf("0.99") if lam > 1 else 1 - ctx.mpf("0.02") * lam + ctx.mpf("0.01") * lam**2)
    )
    u = ((ft - fr_tilde) / fr_tilde) ** 2
    x_nd = u - ctx.mpf("0.571505") * c - ctx.mpf("0.00508451") * chi
    x_prime_nd = u - ctx.mpf("0.657424") * c - ctx.mpf("0.0259977") * chi
    x_d = t + ctx.mpf("0.424912") * c + ctx.mpf("0.363604") * ctx.sqrt(eta) - ctx.mpf("0.060559") * chi
    x_prime_d = (
        t
        - ctx.mpf("0.132754") * c
        + ctx.mpf("0.576669") * ctx.sqrt(eta)
        - ctx.mpf("0.0603749") * chi
        - ctx.mpf("0.0601185") * chi**2
        - ctx.mpf("0.0729134") * chi**3
    )

    def window_plus(x, x0, width):
        return (1 + ctx.tanh(4 * (x - ctx.mpf(x0)) / ctx.mpf(width))) / 2

    def window_minus(x, x0, width):
        return (1 - ctx.tanh(4 * (x - ctx.mpf(x0)) / ctx.mpf(width))) / 2

    eps_high = window_plus(x_nd, "-0.0796251", "0.0801192")
    sig_high = window_minus(x_prime_nd, "-0.206465", "0.226844")
    eps_disrupt = ctx.mpf("1.29971") - ctx.mpf("1.61724") * x_d
    sig_disrupt = ctx.mpf("0.137722") - ctx.mpf("0.293237") * x_prime_d
    high = ft >= fr
    has_torus = t > 0
    eps_tide = eps_high if high else ctx.zero
    eps_ins = ctx.one if high and not has_torus else eps_disrupt
    sigma_tide = sig_high if high else sig_disrupt if has_torus else (sig_high + sig_disrupt) / 2
    return BranchAmplitudeFits(fr_tilde, x_nd, x_prime_nd, x_d, x_prime_d, eps_tide, eps_ins, sigma_tide)
