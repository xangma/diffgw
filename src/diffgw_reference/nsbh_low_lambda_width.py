"""Independent scalar Thompson Eq. (29) low-Lambda / Eq. (28) reference."""

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class LowLambdaWidth:
    delta2_prime: mpmath.mpf
    sigma: mpmath.mpf
    d_delta2_prime_d_lambda: mpmath.mpf
    d_delta2_prime_d_delta2: mpmath.mpf
    d_sigma_d_lambda: mpmath.mpf
    d_sigma_d_delta2: mpmath.mpf
    d_sigma_d_f_rd: mpmath.mpf
    d_sigma_d_q: mpmath.mpf


def low_lambda_ringdown_width(delta2, lambda_ns, f_rd, quality_factor, *, dps=70):
    """Evaluate the printed closed low-tidal branch with supplied inputs."""
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    delta, lam, frequency, quality = (
        ctx.mpf(str(value)) for value in (delta2, lambda_ns, f_rd, quality_factor)
    )
    if not all(ctx.isfinite(value) for value in (delta, lam, frequency, quality)):
        raise ValueError("Expected finite low-Lambda width inputs")
    if delta < ctx.mpf("1e-4") or not 0 <= lam <= 1 or frequency <= 0 or quality <= 0:
        raise ValueError("Expected delta2>=1e-4, 0<=Lambda<=1, f_RD>0, Q>0")

    b0 = ctx.mpf("0.81248")
    correction = delta - b0
    delta_prime = delta - 2 * correction * lam + correction * lam**2
    sigma = delta_prime * frequency / quality
    d_prime_d_lam = 2 * correction * (lam - 1)
    d_prime_d_delta = (1 - lam) ** 2
    return LowLambdaWidth(
        delta2_prime=delta_prime,
        sigma=sigma,
        d_delta2_prime_d_lambda=d_prime_d_lam,
        d_delta2_prime_d_delta2=d_prime_d_delta,
        d_sigma_d_lambda=d_prime_d_lam * frequency / quality,
        d_sigma_d_delta2=d_prime_d_delta * frequency / quality,
        d_sigma_d_f_rd=delta_prime / quality,
        d_sigma_d_q=-delta_prime * frequency / quality**2,
    )
