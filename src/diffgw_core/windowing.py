"""Differentiable windowing functions and tapers for gravitational waveforms.

Provides infinitely differentiable (C-infinity) smooth windowing layers (Planck taper)
with pure Python/NumPy, PyTorch, and JAX implementations.
"""

from __future__ import annotations

import math
from typing import Any, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

__all__ = [
    "planck_taper",
    "torch_planck_taper",
    "jax_planck_taper",
    "nsbh_gamma1_continuation",
    "torch_nsbh_gamma1_continuation",
    "jax_nsbh_gamma1_continuation",
    "nsbh_ringdown_width_window",
    "torch_nsbh_ringdown_width_window",
    "jax_nsbh_ringdown_width_window",
]


def planck_taper(
    f: Union[float, int, Any],
    f_start: float,
    f_end: float,
) -> Union[float, Any]:
    """Evaluate smooth Planck-taper window function over [f_start, f_end].

    For a roll-off (falling) window:
      w(f) = 1.0                                      for f <= f_start
      w(f) = 1.0 / (exp(sigma(f)) + 1.0)              for f_start < f < f_end
      w(f) = 0.0                                      for f >= f_end

    where sigma(f) = (f_end - f_start)/(f_end - f) - (f_end - f_start)/(f - f_start).

    Args:
        f: Evaluation frequency (scalar float/int or numpy array/iterable).
        f_start: Frequency at which tapering starts (w=1 for f <= f_start).
        f_end: Frequency at which tapering ends (w=0 for f >= f_end).

    Returns:
        Taper window values with matching scalar or array format.
    """
    if f_start >= f_end:
        raise ValueError(f"f_start ({f_start}) must be strictly less than f_end ({f_end})")

    # Scalar float or int
    if isinstance(f, (int, float)) or (np is not None and np.isscalar(f)):
        f_val = float(f)
        if f_val <= f_start:
            return 1.0
        if f_val >= f_end:
            return 0.0
        delta = f_end - f_start
        z = (f_val - f_start) / delta
        inv = (1.0 / (1.0 - z)) - (1.0 / z)
        if inv > 700.0:
            return 0.0
        if inv < -700.0:
            return 1.0
        return 1.0 / (math.exp(inv) + 1.0)

    # Numpy array or iterable
    if np is None:
        return [planck_taper(x, f_start, f_end) for x in f]

    f_arr = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f_arr)
    mask_left = f_arr <= f_start
    mask_right = f_arr >= f_end
    mask_mid = (~mask_left) & (~mask_right)

    out[mask_left] = 1.0
    out[mask_right] = 0.0

    if np.any(mask_mid):
        f_mid = f_arr[mask_mid]
        delta = f_end - f_start
        z = (f_mid - f_start) / delta
        inv = (1.0 / (1.0 - z)) - (1.0 / z)
        inv_clipped = np.clip(inv, -700.0, 700.0)
        out[mask_mid] = 1.0 / (np.exp(inv_clipped) + 1.0)

    return out


def torch_planck_taper(
    f: torch.Tensor,
    f_start: Union[float, torch.Tensor],
    f_end: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Evaluate smooth Planck-taper window in PyTorch with stable autodiff.

    Computes:
      w(f) = 1.0                                      for f <= f_start
      w(f) = 1.0 / (exp(sigma(f)) + 1.0)              for f_start < f < f_end
      w(f) = 0.0                                      for f >= f_end
    where sigma(f) = (f_end - f_start)/(f_end - f) - (f_end - f_start)/(f - f_start).

    Guarantees no NaN or Inf in forward evaluation or autograd backward passes
    for all f values (including boundaries and out-of-bounds regions) via
    the double-where trick and stable clipping.

    Args:
        f: Frequency tensor.
        f_start: Taper start frequency (scalar or broadcastable tensor).
        f_end: Taper end frequency (scalar or broadcastable tensor).

    Returns:
        Window tensor matching device, shape, and floating dtype of f.
    """
    if torch is None:
        raise ImportError("PyTorch is required for torch_planck_taper. Install torch.")

    if not isinstance(f, torch.Tensor):
        f = torch.as_tensor(f)

    device = f.device
    dtype = f.dtype if f.is_floating_point() else torch.float64
    if f.dtype != dtype:
        f = f.to(dtype=dtype)

    f_start_t = torch.as_tensor(f_start, dtype=dtype, device=device)
    f_end_t = torch.as_tensor(f_end, dtype=dtype, device=device)

    delta = f_end_t - f_start_t
    delta_safe = torch.where(delta == 0, torch.ones_like(delta), delta)

    in_bounds = (f > f_start_t) & (f < f_end_t)
    z = (f - f_start_t) / delta_safe
    # Dummy branch inside (0, 1) prevents NaN/inf in both forward and backward passes
    z_safe = torch.where(in_bounds, z, torch.full_like(z, 0.5))

    eps = 1e-15 if dtype == torch.float64 else 1e-7
    z_clamped = torch.clamp(z_safe, eps, 1.0 - eps)

    inv = (1.0 / (1.0 - z_clamped)) - (1.0 / z_clamped)
    max_exp = 100.0 if dtype == torch.float64 else 50.0
    inv_clipped = torch.clamp(inv, -max_exp, max_exp)
    w_smooth = 1.0 / (torch.exp(inv_clipped) + 1.0)

    w = torch.where(f <= f_start_t, torch.ones_like(w_smooth), w_smooth)
    w = torch.where(f >= f_end_t, torch.zeros_like(w), w)
    return w


def jax_planck_taper(
    f: jnp.ndarray,
    f_start: Union[float, jnp.ndarray],
    f_end: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Evaluate smooth Planck-taper window in JAX with JIT and autodiff compatibility.

    Computes:
      w(f) = 1.0                                      for f <= f_start
      w(f) = 1.0 / (exp(sigma(f)) + 1.0)              for f_start < f < f_end
      w(f) = 0.0                                      for f >= f_end
    where sigma(f) = (f_end - f_start)/(f_end - f) - (f_end - f_start)/(f - f_start).

    Fully compatible with jax.jit, jax.grad, jax.vmap, and reverse-mode autodiff.

    Args:
        f: Frequency array.
        f_start: Taper start frequency (scalar or broadcastable array).
        f_end: Taper end frequency (scalar or broadcastable array).

    Returns:
        Window array matching floating dtype and shape of f.
    """
    if jax is None or jnp is None:
        raise ImportError("JAX is required for jax_planck_taper. Install jax.")

    dtype = f.dtype if jnp.issubdtype(f.dtype, jnp.floating) else jnp.float64
    if f.dtype != dtype:
        f = f.astype(dtype)

    f_start_t = jnp.asarray(f_start, dtype=dtype)
    f_end_t = jnp.asarray(f_end, dtype=dtype)

    delta = f_end_t - f_start_t
    delta_safe = jnp.where(delta == 0, 1.0, delta)

    in_bounds = (f > f_start_t) & (f < f_end_t)
    z = (f - f_start_t) / delta_safe
    z_safe = jnp.where(in_bounds, z, 0.5)

    eps = 1e-15 if dtype == jnp.float64 else 1e-7
    z_clamped = jnp.clip(z_safe, eps, 1.0 - eps)

    inv = (1.0 / (1.0 - z_clamped)) - (1.0 / z_clamped)
    max_exp = 100.0 if dtype == jnp.float64 else 50.0
    inv_clipped = jnp.clip(inv, -max_exp, max_exp)
    w_smooth = 1.0 / (jnp.exp(inv_clipped) + 1.0)

    w = jnp.where(f <= f_start_t, 1.0, w_smooth)
    w = jnp.where(f >= f_end_t, 0.0, w)
    return w


# ---------------------------------------------------------------------------
# IMRPhenomNSBH Amplitude Continuation and Window Functions
# ---------------------------------------------------------------------------

def nsbh_gamma1_continuation(
    lambda_ns: Union[float, int, Any],
    variant: str = "project_correction",
) -> Union[float, Any]:
    r"""Evaluate the premerger tidal correction factor gamma'_1 / gamma_1 for IMRPhenomNSBH.

    Thompson et al. (2020), Phys. Rev. D 101, 124059, Eq. (26) defines the tidal
    correction factor for premerger amplitude coefficient gamma_1.

    In the published paper, Eq. (26) literally prints:
      \gamma'_1 / \gamma_1 = 1 - 0.5*\Lambda - 0.25*\Lambda^2   for \Lambda <= 1
      \gamma'_1 / \gamma_1 = 1.25                               for \Lambda > 1

    Notice that at \Lambda = 1, the printed expression evaluates to:
      1 - 0.5(1) - 0.25(1)^2 = 0.25
    whereas for \Lambda > 1, it is 1.25. This creates an unphysical jump
    discontinuity of \Delta = +1.0 at \Lambda = 1.

    The smooth polynomial continuation variant (labeled 'project_correction' or
    'compatibility') resolves this discrepancy via:
      g(\Lambda) = 1 + 0.5*\Lambda - 0.25*\Lambda^2             for \Lambda <= 1
      g(\Lambda) = 1.25                                         for \Lambda > 1

    Properties of the 'project_correction' variant:
      1. Smooth BBH connection: at \Lambda = 0, g(0) = 1.0, recovering the BBH
         baseline amplitude exactly without jump.
      2. Boundary continuity: at \Lambda = 1, g(1) = 1 + 0.5(1) - 0.25(1)^2 = 1.25,
         matching the \Lambda > 1 constant value.
      3. First-derivative match: g'(\Lambda) = 0.5 - 0.5*\Lambda, which vanishes
         at \Lambda = 1 (g'(1) = 0), matching the zero derivative of the constant
         1.25 for \Lambda > 1, ensuring C^1 smoothness across the boundary.

    Args:
        lambda_ns: Dimensionless tidal deformability Lambda (scalar or array).
        variant: 'project_correction' (default, smooth C^1 continuation) or
                 'paper' (published paper Eq. 26 with jump at Lambda=1).

    Returns:
        Correction factor gamma'_1 / gamma_1 matching input shape and type.
    """
    if isinstance(lambda_ns, (int, float)):
        lam = max(0.0, float(lambda_ns))
        if lam > 1.0:
            return 1.25
        if variant in ("project_correction", "compatibility", "smooth"):
            return 1.0 + 0.5 * lam - 0.25 * (lam ** 2)
        elif variant in ("paper", "published"):
            return 1.0 - 0.5 * lam - 0.25 * (lam ** 2)
        else:
            raise ValueError(f"Unknown variant '{variant}', expected 'project_correction' or 'paper'")

    lam_arr = np.asarray(lambda_ns, dtype=np.float64) if np is not None else lambda_ns
    lam_safe = np.maximum(lam_arr, 0.0)
    if variant in ("project_correction", "compatibility", "smooth"):
        small_poly = 1.0 + 0.5 * lam_safe - 0.25 * (lam_safe ** 2)
    elif variant in ("paper", "published"):
        small_poly = 1.0 - 0.5 * lam_safe - 0.25 * (lam_safe ** 2)
    else:
        raise ValueError(f"Unknown variant '{variant}', expected 'project_correction' or 'paper'")

    return np.where(lam_safe > 1.0, 1.25, small_poly)


def torch_nsbh_gamma1_continuation(
    lambda_ns: torch.Tensor,
    variant: str = "project_correction",
) -> torch.Tensor:
    r"""Evaluate the premerger tidal correction factor in PyTorch with autograd support.

    Computes gamma'_1 / gamma_1 with smooth C^1 polynomial continuation across Lambda=1
    by default ('project_correction' variant), or the published Eq. (26) paper variant.
    """
    if torch is None:
        raise ImportError("PyTorch is required for torch_nsbh_gamma1_continuation.")

    lam_safe = torch.clamp(lambda_ns, min=0.0)
    if variant in ("project_correction", "compatibility", "smooth"):
        small_poly = 1.0 + 0.5 * lam_safe - 0.25 * (lam_safe ** 2.0)
    elif variant in ("paper", "published"):
        small_poly = 1.0 - 0.5 * lam_safe - 0.25 * (lam_safe ** 2.0)
    else:
        raise ValueError(f"Unknown variant '{variant}', expected 'project_correction' or 'paper'")

    return torch.where(lambda_ns > 1.0, torch.full_like(lambda_ns, 1.25), small_poly)


def jax_nsbh_gamma1_continuation(
    lambda_ns: jnp.ndarray,
    variant: str = "project_correction",
) -> jnp.ndarray:
    r"""Evaluate the premerger tidal correction factor in JAX with autodiff support.

    Computes gamma'_1 / gamma_1 with smooth C^1 polynomial continuation across Lambda=1
    by default ('project_correction' variant), or the published Eq. (26) paper variant.
    """
    if jax is None or jnp is None:
        raise ImportError("JAX is required for jax_nsbh_gamma1_continuation.")

    lam_safe = jnp.maximum(lambda_ns, 0.0)
    if variant in ("project_correction", "compatibility", "smooth"):
        small_poly = 1.0 + 0.5 * lam_safe - 0.25 * (lam_safe ** 2.0)
    elif variant in ("paper", "published"):
        small_poly = 1.0 - 0.5 * lam_safe - 0.25 * (lam_safe ** 2.0)
    else:
        raise ValueError(f"Unknown variant '{variant}', expected 'project_correction' or 'paper'")

    return jnp.where(lambda_ns > 1.0, 1.25, small_poly)


def nsbh_ringdown_width_window(
    f_tide: Union[float, Any],
    f_rd_tilde: Union[float, Any],
    variant: str = "project_correction",
) -> Union[float, Any]:
    r"""Evaluate the large-tidal-frequency ringdown width window delta'_2 for IMRPhenomNSBH.

    Thompson et al. (2020), Eq. (29), defines delta'_2 for Lambda > 1:
      \delta'_2 = (A/2) * \omega^-_{x_3, d_3}(x)
    with x = (f_tide - f_rd_tilde) / f_rd_tilde, x_3 = 0.0188092, d_3 = 0.338737,
    and A/2 = 0.81248. The window function is defined in Eq. (30) as:
      \omega^\pm_{x_0, d}(x) = [1 \pm \tanh(4*(x - x_0) / d)] / 2

    Sign Convention & Asymptotic Limits:
      - The published paper Eq. (29) prints \omega^- (minus window):
          0.81248 * 0.5 * (1 - \tanh(...))
        In the large tidal disruption frequency limit (f_tide >> f_rd_tilde, e.g.
        for compact NS or BBH where disruption does not occur early), x -> +\infty,
        so \tanh -> +1 and 1 - \tanh -> 0. This causes delta'_2 to unphysically
        collapse to 0.
      - The 'project_correction' variant (+0.81248 \omega^+ plus window):
          0.81248 * 0.5 * (1 + \tanh(...))
        As x -> +\infty, \tanh -> +1 and 1 + \tanh -> 2, so the window factor smoothly
        approaches 0.81248, matching the BBH ringdown width constant \delta_2 = 0.81248
        and ensuring continuity across disruption regimes.

    Args:
        f_tide: Tidal disruption frequency in Hz.
        f_rd_tilde: Adjusted ringdown frequency in Hz.
        variant: 'project_correction' / 'plus_window' (+0.81248 \omega^+, default)
                 or 'paper' / 'minus_window' (published Eq. 29 minus window).

    Returns:
        Ringdown width window factor delta'_2.
    """
    x3 = 0.0188092
    d3 = 0.338737
    scale = 0.81248 * 0.5

    if isinstance(f_tide, (int, float)) and isinstance(f_rd_tilde, (int, float)):
        x = (float(f_tide) - float(f_rd_tilde)) / float(f_rd_tilde)
        arg = 4.0 * (x - x3) / d3
        tanh_val = math.tanh(arg)
        if variant in ("project_correction", "plus_window", "+", "compatibility"):
            return scale * (1.0 + tanh_val)
        elif variant in ("paper", "minus_window", "-"):
            return scale * (1.0 - tanh_val)
        else:
            raise ValueError(f"Unknown variant '{variant}', expected 'project_correction' or 'paper'")

    ft = np.asarray(f_tide, dtype=np.float64)
    fr = np.asarray(f_rd_tilde, dtype=np.float64)
    x = (ft - fr) / fr
    arg = 4.0 * (x - x3) / d3
    tanh_val = np.tanh(arg)

    if variant in ("project_correction", "plus_window", "+", "compatibility"):
        return scale * (1.0 + tanh_val)
    elif variant in ("paper", "minus_window", "-"):
        return scale * (1.0 - tanh_val)
    else:
        raise ValueError(f"Unknown variant '{variant}', expected 'project_correction' or 'paper'")


def torch_nsbh_ringdown_width_window(
    f_tide: torch.Tensor,
    f_rd_tilde: torch.Tensor,
    variant: str = "project_correction",
) -> torch.Tensor:
    """Evaluate NSBH ringdown width window in PyTorch with autograd support."""
    if torch is None:
        raise ImportError("PyTorch is required for torch_nsbh_ringdown_width_window.")

    x3 = 0.0188092
    d3 = 0.338737
    scale = 0.81248 * 0.5
    x = (f_tide - f_rd_tilde) / f_rd_tilde
    arg = 4.0 * (x - x3) / d3
    tanh_val = torch.tanh(arg)

    if variant in ("project_correction", "plus_window", "+", "compatibility"):
        return scale * (1.0 + tanh_val)
    elif variant in ("paper", "minus_window", "-"):
        return scale * (1.0 - tanh_val)
    else:
        raise ValueError(f"Unknown variant '{variant}', expected 'project_correction' or 'paper'")


def jax_nsbh_ringdown_width_window(
    f_tide: jnp.ndarray,
    f_rd_tilde: jnp.ndarray,
    variant: str = "project_correction",
) -> jnp.ndarray:
    """Evaluate NSBH ringdown width window in JAX with autodiff support."""
    if jax is None or jnp is None:
        raise ImportError("JAX is required for jax_nsbh_ringdown_width_window.")

    x3 = 0.0188092
    d3 = 0.338737
    scale = 0.81248 * 0.5
    x = (f_tide - f_rd_tilde) / f_rd_tilde
    arg = 4.0 * (x - x3) / d3
    tanh_val = jnp.tanh(arg)

    if variant in ("project_correction", "plus_window", "+", "compatibility"):
        return scale * (1.0 + tanh_val)
    elif variant in ("paper", "minus_window", "-"):
        return scale * (1.0 - tanh_val)
    else:
        raise ValueError(f"Unknown variant '{variant}', expected 'project_correction' or 'paper'")
