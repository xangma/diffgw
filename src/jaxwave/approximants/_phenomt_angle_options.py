"""Backend-neutral Appendix A angle continuations for JAX PhenomT models."""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp

from diffgw_core.phenomt_options import resolve_precession


def canonical_precession_version(method: str, version=None) -> int:
    return resolve_precession(method, version)[1]


def minimal_rotation_gamma(
    times,
    angle_evaluator: Callable,
    gamma_at_reference,
    *,
    reference_time,
    n_grid: int = 1024,
):
    u = jnp.linspace(0.0, 1.0, n_grid, dtype=times.dtype)
    grid = reference_time[..., None, None] + (times[..., None] - reference_time[..., None, None]) * u
    alpha, beta = angle_evaluator(grid)
    dt = (times - reference_time[..., None]) / (n_grid - 1)
    eps = jnp.finfo(times.dtype).eps
    dt_safe = jnp.where(jnp.abs(dt) < eps, eps, dt)
    adot = (alpha[..., 1:] - alpha[..., :-1]) / dt_safe[..., None]
    gdot = -jnp.cos(0.5 * (beta[..., 1:] + beta[..., :-1])) * adot
    return gamma_at_reference[..., None] + jnp.sum(gdot * dt[..., None], axis=-1)


def linear_meco_continuation(
    times,
    alpha,
    beta,
    meco_time,
    *,
    angle_evaluator: Optional[Callable] = None,
    n_grid: int = 1024,
):
    del n_grid
    if angle_evaluator is None:
        raise ValueError("MECO continuation requires angle_evaluator")
    dt = jnp.maximum(jnp.abs(meco_time) * 1e-4, jnp.finfo(times.dtype).eps)
    ap, bp = angle_evaluator(meco_time[..., None] + dt[..., None])
    am_, bm_ = angle_evaluator(meco_time[..., None] - dt[..., None])
    am, bm = angle_evaluator(meco_time[..., None])
    am, bm = am[..., 0], bm[..., 0]
    da = (ap[..., 0] - am_[..., 0]) / (2.0 * dt)
    db = (bp[..., 0] - bm_[..., 0]) / (2.0 * dt)
    continuation = times >= meco_time[..., None]
    a_lin = am[..., None] + da[..., None] * (times - meco_time[..., None])
    b_lin = bm[..., None] + db[..., None] * (times - meco_time[..., None])
    return jnp.where(continuation, a_lin, alpha), jnp.where(continuation, b_lin, beta)


def apply_angle_option(
    times, alpha, beta, gamma, alpha_peak, beta_peak, gamma_peak, slope, precession_version,
    *, angle_evaluator: Optional[Callable] = None, meco_time=None, reference_time=None,
    gamma_reference=None, n_grid: int = 1024,
):
    """Apply Appendix A suffix digits to raw NNLO or MSA angles."""
    suffix = int(precession_version) % 100
    if suffix == 0:
        return alpha, beta, gamma
    if suffix == 21:
        if angle_evaluator is None or meco_time is None:
            raise ValueError("angle option 21 requires angle_evaluator and meco_time")
        alpha, beta = linear_meco_continuation(
            times, alpha, beta, meco_time, angle_evaluator=angle_evaluator, n_grid=n_grid
        )
        zero = jnp.zeros_like(times[..., :1])
        alpha_peak, beta_peak = linear_meco_continuation(
            zero, alpha, beta, meco_time, angle_evaluator=angle_evaluator, n_grid=n_grid
        )
        alpha_peak, beta_peak = alpha_peak[..., 0], beta_peak[..., 0]
    gamma_evaluator = angle_evaluator
    if suffix == 21:
        # The numerical gamma integral must use the linearly continued
        # alpha/beta trajectory after MECO, rather than the raw analytical
        # evaluator used below MECO.
        dt = jnp.maximum(jnp.abs(meco_time) * 1e-4, jnp.finfo(times.dtype).eps)
        ap, bp = angle_evaluator(meco_time[..., None] + dt[..., None])
        am_, bm_ = angle_evaluator(meco_time[..., None] - dt[..., None])
        av, bv = angle_evaluator(meco_time[..., None])
        av, bv = av[..., 0], bv[..., 0]
        da = (ap[..., 0] - am_[..., 0]) / (2.0 * dt)
        db = (bp[..., 0] - bm_[..., 0]) / (2.0 * dt)

        def gamma_evaluator(query):
            raw_a, raw_b = angle_evaluator(query)

            def expand(value):
                return value.reshape(value.shape + (1,) * (query.ndim - value.ndim))

            mt = expand(meco_time)
            a_lin = expand(av) + expand(da) * (query - mt)
            b_lin = expand(bv) + expand(db) * (query - mt)
            use_linear = query >= mt
            return jnp.where(use_linear, a_lin, raw_a), jnp.where(use_linear, b_lin, raw_b)
    if suffix in (11, 21):
        if angle_evaluator is None or reference_time is None or gamma_reference is None:
            raise ValueError("numerical gamma options require angle_evaluator, reference_time, and gamma_reference")
        gamma = minimal_rotation_gamma(
            times, gamma_evaluator, gamma_reference, reference_time=reference_time, n_grid=n_grid
        )
        gamma_peak = minimal_rotation_gamma(
            jnp.zeros_like(times[..., :1]), gamma_evaluator, gamma_reference,
            reference_time=reference_time, n_grid=n_grid,
        )[..., 0]
    post = times >= 0
    rd_alpha = alpha_peak[..., None] + slope[..., None] * times
    rd_gamma = gamma_peak[..., None] - jnp.cos(beta_peak)[..., None] * (rd_alpha - alpha_peak[..., None])
    return jnp.where(post, rd_alpha, alpha), jnp.where(post, beta_peak[..., None], beta), jnp.where(post, rd_gamma, gamma)


__all__ = ["canonical_precession_version", "minimal_rotation_gamma", "linear_meco_continuation", "apply_angle_option"]
