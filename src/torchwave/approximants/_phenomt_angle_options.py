"""Published PhenomT angle-continuation options.

The helpers in this module deliberately operate on angles rather than on a
particular precession model.  This lets the NNLO and MSA evaluators share the
Appendix A continuation rules while keeping all operations differentiable.
``angle_evaluator`` is called on a fixed internal grid; callers may therefore
use a carrier-frequency evaluator instead of making the requested waveform
grid part of the integration rule.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from diffgw_core.phenomt_options import resolve_precession


def canonical_precession_version(method: str, version=None) -> int:
    """Return the validated Appendix A integer for a backend evaluator."""
    return resolve_precession(method, version)[1]


def minimal_rotation_gamma(
    times: torch.Tensor,
    angle_evaluator: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    gamma_at_reference: torch.Tensor,
    *,
    reference_time: torch.Tensor,
    n_grid: int = 1024,
) -> torch.Tensor:
    """Integrate ``gamma_dot = -alpha_dot cos(beta)`` on a fixed grid.

    Each output time gets its own fixed quadrature interval from the supplied
    reference time.  The requested waveform grid consequently never changes
    the numerical integration nodes.
    """
    u = torch.linspace(0.0, 1.0, n_grid, dtype=times.dtype, device=times.device)
    grid = reference_time[..., None, None] + (times[..., None] - reference_time[..., None, None]) * u
    alpha, beta = angle_evaluator(grid)
    dt = (times - reference_time[..., None]) / (n_grid - 1)
    eps = torch.finfo(times.dtype).eps
    dt_safe = torch.where(dt.abs() < eps, torch.full_like(dt, eps), dt)
    adot = (alpha[..., 1:] - alpha[..., :-1]) / dt_safe[..., None]
    gdot = -torch.cos(0.5 * (beta[..., 1:] + beta[..., :-1])) * adot
    increments = gdot * dt[..., None]
    return gamma_at_reference[..., None] + increments.sum(dim=-1)


def linear_meco_continuation(
    times: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    meco_time: torch.Tensor,
    *,
    angle_evaluator: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
    n_grid: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continue alpha and beta linearly from MECO to ``t=0`` (Eqs. 20a-b)."""
    if angle_evaluator is None:
        raise ValueError("MECO continuation requires angle_evaluator")
    dt = torch.clamp(meco_time.abs() * 1e-4, min=torch.finfo(times.dtype).eps)
    ap, bp = angle_evaluator(meco_time[..., None] + dt[..., None])
    am_, bm_ = angle_evaluator(meco_time[..., None] - dt[..., None])
    am, bm = angle_evaluator(meco_time[..., None])
    am, bm = am[..., 0], bm[..., 0]
    da = (ap[..., 0] - am_[..., 0]) / (2.0 * dt)
    db = (bp[..., 0] - bm_[..., 0]) / (2.0 * dt)
    continuation = times >= meco_time[..., None]
    a_lin = am[..., None] + da[..., None] * (times - meco_time[..., None])
    b_lin = bm[..., None] + db[..., None] * (times - meco_time[..., None])
    return torch.where(continuation, a_lin, alpha), torch.where(continuation, b_lin, beta)


def apply_angle_option(
    times: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    alpha_peak: torch.Tensor,
    beta_peak: torch.Tensor,
    gamma_peak: torch.Tensor,
    slope: torch.Tensor,
    precession_version: int,
    *,
    angle_evaluator: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
    meco_time: Optional[torch.Tensor] = None,
    reference_time: Optional[torch.Tensor] = None,
    gamma_reference: Optional[torch.Tensor] = None,
    n_grid: int = 1024,
):
    """Apply Appendix A suffix digits to raw inspiral angles.

    ``alpha_evaluator`` returns raw alpha and beta at arbitrary times.  The
    first three arguments are accepted from either NNLO or MSA evaluators.
    """
    suffix = int(precession_version) % 100
    gamma_evaluator = angle_evaluator
    if suffix == 0:
        return alpha, beta, gamma
    if suffix == 21:
        if angle_evaluator is None or meco_time is None:
            raise ValueError("angle option 21 requires angle_evaluator and meco_time")
        alpha, beta = linear_meco_continuation(
            times, alpha, beta, meco_time, angle_evaluator=angle_evaluator, n_grid=n_grid
        )
        zero = torch.zeros_like(times[..., :1])
        alpha_peak, beta_peak = linear_meco_continuation(
            zero, alpha, beta, meco_time, angle_evaluator=angle_evaluator, n_grid=n_grid
        )
        alpha_peak, beta_peak = alpha_peak[..., 0], beta_peak[..., 0]
        # The numerical gamma in 21 follows the continued alpha and beta,
        # rather than silently reverting to the raw PN angles after MECO.
        dt = torch.clamp(meco_time.abs() * 1e-4, min=torch.finfo(times.dtype).eps)
        ap, bp = angle_evaluator(meco_time[..., None] + dt[..., None])
        am, bm = angle_evaluator(meco_time[..., None] - dt[..., None])
        av, bv = angle_evaluator(meco_time[..., None])
        av, bv = av[..., 0], bv[..., 0]
        da = (ap[..., 0] - am[..., 0]) / (2 * dt)
        db = (bp[..., 0] - bm[..., 0]) / (2 * dt)

        def gamma_evaluator(query):
            raw_a, raw_b = angle_evaluator(query)
            def expand(value):
                return value.reshape(value.shape + (1,) * (query.ndim - value.ndim))
            mt = expand(meco_time)
            aa = expand(av) + expand(da) * (query - mt)
            bb = expand(bv) + expand(db) * (query - mt)
            use_linear = query >= mt
            return torch.where(use_linear, aa, raw_a), torch.where(use_linear, bb, raw_b)
    if suffix == 11 or suffix == 21:
        if angle_evaluator is None or reference_time is None or gamma_reference is None:
            raise ValueError("numerical gamma options require angle_evaluator, reference_time, and gamma_reference")
        gamma = minimal_rotation_gamma(
            times, gamma_evaluator, gamma_reference, reference_time=reference_time, n_grid=n_grid
        )
        gamma_peak = minimal_rotation_gamma(
            torch.zeros_like(times[..., :1]), gamma_evaluator, gamma_reference,
            reference_time=reference_time, n_grid=n_grid,
        )[..., 0]
    post = times >= 0
    rd_alpha = alpha_peak[..., None] + slope[..., None] * times
    rd_gamma = gamma_peak[..., None] - torch.cos(beta_peak)[..., None] * (rd_alpha - alpha_peak[..., None])
    return torch.where(post, rd_alpha, alpha), torch.where(post, beta_peak[..., None], beta), torch.where(post, rd_gamma, gamma)


__all__ = [
    "canonical_precession_version",
    "minimal_rotation_gamma",
    "linear_meco_continuation",
    "apply_angle_option",
]
