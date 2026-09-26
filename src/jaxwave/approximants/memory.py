"""Provisional strain-memory carrier and a published dominant-(2, ±2) slice.

The legacy ``compute_memory_h20`` is a diagnostic energy-flux proxy. The
explicit ``*_22`` path follows Favata (2010), arXiv:1003.3486v1, Sec. 6.2,
and Talbot, Thrane, Lasky & Lin (2018), arXiv:1807.00990v2, Eqs. (1)-(9).
"""

from typing import Dict, Optional, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    G_SI,
    MPC_SI,
    PI,
)
from jaxwave.conversions import as_array


# Spherical harmonic coefficient for (l=2, m=0): sqrt(15 / (32 * pi))
Y20_COEFF: float = math.sqrt(15.0 / (32.0 * math.pi))


def compute_memory_h20_22(
    times: jnp.ndarray,
    h22: jnp.ndarray,
    distance_m: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Published ±22-sourced h20 with explicit SI distance and sample-zero offset.

    See the Torch counterpart for the normalization and convention. The
    supplied complex h22 is physical strain; times are seconds, distance_m
    is meters, and the time grid may be nonuniform.
    """
    if times.ndim != 1 or times.size < 2 or h22.shape[-1] != times.size:
        raise ValueError("times must be 1D with at least two samples and match h22")
    if isinstance(distance_m, (float, int)) and distance_m <= 0:
        raise ValueError("distance_m must be positive")
    slope = jnp.gradient(h22, times, axis=-1)
    flux = jnp.abs(slope) ** 2
    increments = (flux[..., :-1] + flux[..., 1:]) * jnp.diff(times) / 2
    integral = jnp.concatenate((jnp.zeros_like(increments[..., :1]), jnp.cumsum(increments, axis=-1)), axis=-1)
    distance = as_array(distance_m, dtype=times.dtype)
    return distance * (math.sqrt(5.0 / (6.0 * math.pi)) / (7.0 * C_SI)) * integral


def memory_polarizations_22(
    h20_mem: jnp.ndarray,
    inclination: Union[float, jnp.ndarray] = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Full dominant-22 pattern, including the l=4,m=0 memory projection."""
    inc = as_array(inclination, dtype=h20_mem.dtype)
    cos_i = jnp.cos(inc)
    pattern = Y20_COEFF * jnp.sin(inc) ** 2 * (7.0 / 120.0) * (17.0 + cos_i ** 2)
    if h20_mem.ndim > pattern.ndim:
        pattern = jnp.expand_dims(pattern, -1)
    hp_mem = h20_mem * pattern
    return hp_mem, jnp.zeros_like(hp_mem)


def compute_memory_h20(
    times: jnp.ndarray,
    modes_or_hp: Union[jnp.ndarray, Dict[Tuple[int, int], jnp.ndarray]],
    hc: Optional[jnp.ndarray] = None,
    distance: Optional[Union[float, jnp.ndarray]] = None,
    scale_factor: float = 1.0,
) -> jnp.ndarray:
    """Compute the non-linear Christodoulou memory mode h_20^mem(t).

    Formula:
        h_20^mem(t) = (1 / sqrt(6)) * int_{-infty}^t dt' [ sum_{l, m} |dh_{lm}/dt'|^2 ] * prefactor
    References:
        Favata (2010) Eq. (2.12)-(2.13); Talbot et al. (2018) Eq. (7)

    Guaranteed properties:
        1. Monotonically increasing: d(h_20^mem)/dt >= 0 (since integrand is sum of squared terms >= 0).
        2. Constant post-merger saturation: h_20^mem(t) -> Delta h^mem as t -> infty.

    Args:
        times: Time array in seconds, shape (N,)
        modes_or_hp: Either a dictionary of modes {(l, m): h_lm(t)} or time-domain h_plus(t)
        hc: Time-domain h_cross(t) if modes_or_hp is h_plus
        distance: Luminosity distance in meters (optional scaling)
        scale_factor: Dimensional or calibration scaling factor

    Returns:
        h20_mem: Memory strain tensor h_20^mem(t), shape (..., N)
    """
    t_tensor = as_array(times, dtype=jnp.float64)
    dt = (t_tensor[1] - t_tensor[0]) if t_tensor.size > 1 else 1.0

    # Build sum of |dh/dt|^2 across available modes
    integrand = jnp.zeros_like(t_tensor)

    if isinstance(modes_or_hp, dict):
        # Multipolar decomposition
        for (l, m), h_lm in modes_or_hp.items():
            h_tensor = as_array(h_lm, dtype=jnp.float64)
            # Centered finite differences for derivative dh/dt
            dh_dt = jnp.gradient(h_tensor, dt, axis=-1)
            integrand = integrand + (jnp.abs(dh_dt) ** 2)
    elif hc is not None:
        # Reconstruct quadrupole energy flux from (h+, hx): |dh22/dt|^2 ~ (dh+/dt)^2 + (dhx/dt)^2
        hp_tensor = as_array(modes_or_hp, dtype=jnp.float64)
        hc_tensor = as_array(hc, dtype=jnp.float64)

        dhp_dt = jnp.gradient(hp_tensor, dt, axis=-1)
        dhc_dt = jnp.gradient(hc_tensor, dt, axis=-1)

        # Flux in dominant (2, 2) and (2, -2) modes
        integrand = (dhp_dt ** 2) + (dhc_dt ** 2)
    else:
        # Single strain series
        h_tensor = as_array(modes_or_hp, dtype=jnp.float64)
        dh_dt = jnp.gradient(h_tensor, dt, axis=-1)
        integrand = dh_dt ** 2

    # Multipolar projection factor 1 / sqrt(6) (Favata 2010, Talbot et al. 2018)
    prefactor = (1.0 / math.sqrt(6.0)) * scale_factor
    if distance is not None:
        dist_t = as_array(distance, dtype=jnp.float64)
        dist_m = jnp.where(dist_t < 1e15, dist_t * MPC_SI, dist_t)
        prefactor = prefactor * (dist_m / (4.0 * PI * C_SI))
    else:
        # Check if input is physical strain (~ 1e-21) or geometric (~ 0.1)
        max_val = jnp.max(integrand)
        fiducial = (100.0 * MPC_SI) / (4.0 * PI * C_SI)
        prefactor = jnp.where((max_val > 0.0) & (max_val < 1e-10), prefactor * fiducial, prefactor)

    # Cumulative trapezoidal integration to guarantee monotonic increase
    # h20_mem(t) = prefactor * int_0^t integrand(t') dt'
    dx = jnp.diff(t_tensor, axis=-1)
    trapz = 0.5 * (integrand[..., :-1] + integrand[..., 1:]) * dx
    cum_trapz = jnp.cumsum(trapz, axis=-1)
    # Pad first element with zero to match length of times
    h20_mem = prefactor * jnp.concatenate([jnp.zeros_like(cum_trapz[..., :1]), cum_trapz], axis=-1)

    return h20_mem


def memory_polarizations(
    h20_mem: jnp.ndarray,
    inclination: Union[float, jnp.ndarray] = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Project memory mode h_20^mem(t) into detector polarizations (h_+^mem, h_x^mem).

    Formulas:
        h_+^mem(t, iota) = h_20^mem(t) * _-2Y_20(iota, 0) = h_20^mem(t) * sqrt(15 / (32 * pi)) * sin^2(iota)
        h_x^mem(t, iota) = 0
    Reference:
        Favata (2010) Eq. (2.14); Talbot et al. (2018) Eq. (8)

    Returns:
        (hp_mem, hc_mem)
    """
    inc = as_array(inclination, dtype=h20_mem.dtype)
    sin_i = jnp.sin(inc)
    y20_pattern = Y20_COEFF * (sin_i * sin_i)

    if h20_mem.ndim > y20_pattern.ndim:
        y20_pattern = jnp.expand_dims(y20_pattern, -1)

    hp_mem = h20_mem * y20_pattern
    hc_mem = jnp.zeros_like(hp_mem)
    return hp_mem, hc_mem


def add_memory_to_waveform(
    times: jnp.ndarray,
    hp: jnp.ndarray,
    hc: jnp.ndarray,
    inclination: Union[float, jnp.ndarray] = 0.0,
    modes_dict: Optional[Dict[Tuple[int, int], jnp.ndarray]] = None,
    distance: Optional[Union[float, jnp.ndarray]] = None,
    scale_factor: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Add non-linear Christodoulou memory strain to oscillatory polarizations.

    Returns:
        (hp_total, hc_total, h20_mem)
    """
    h20 = compute_memory_h20(
        times=times,
        modes_or_hp=modes_dict if modes_dict is not None else hp,
        hc=hc if modes_dict is None else None,
        distance=distance,
        scale_factor=scale_factor,
    )
    hp_mem, hc_mem = memory_polarizations(h20, inclination=inclination)
    hp_total = hp + hp_mem
    hc_total = hc + hc_mem
    return hp_total, hc_total, h20


@jax.tree_util.register_pytree_node_class
class MemoryWaveform:
    """Provisional hp/hc carrier with an explicit published dominant-22 path."""

    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor

    def tree_flatten(self):
        return ((self.scale_factor,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __call__(
        self,
        times: jnp.ndarray,
        hp: jnp.ndarray,
        hc: jnp.ndarray,
        inclination: Union[float, jnp.ndarray] = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return add_memory_to_waveform(
            times=times,
            hp=hp,
            hc=hc,
            inclination=inclination,
            scale_factor=self.scale_factor,
        )

    def evaluate(
        self,
        times: jnp.ndarray,
        hp: jnp.ndarray,
        hc: jnp.ndarray,
        inclination: Union[float, jnp.ndarray] = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self(times=times, hp=hp, hc=hc, inclination=inclination)

    def evaluate_22(
        self,
        times: jnp.ndarray,
        h22: jnp.ndarray,
        distance_m: Union[float, jnp.ndarray],
        inclination: Union[float, jnp.ndarray] = 0.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (h+_mem, hx_mem, h20) for the published ±22-only scope."""
        h20 = compute_memory_h20_22(times, h22, distance_m)
        hp, hc = memory_polarizations_22(h20, inclination)
        return hp, hc, h20
