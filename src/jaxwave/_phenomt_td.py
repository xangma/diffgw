"""Sampling and validation for the native dominant-mode IMRPhenomT API."""

import math
import jax
import jax.numpy as jnp

from .approximants.imrphenomt import IMRPhenomT


def generate_td(
    *,
    mass1,
    mass2,
    chi1,
    chi2,
    spin1z,
    spin2z,
    spin1x,
    spin1y,
    spin2x,
    spin2y,
    distance,
    inclination,
    phic,
    tc,
    lambda1,
    lambda2,
    delta_t,
    f_lower,
    f_final,
    duration,
    t_pad,
    center_coalescence,
    return_times,
    dtype=jnp.float64,
    **kwargs,
):
    """Evaluate on a fixed physical time grid, with no FFT or periodic wrap."""

    def array(value):
        return jnp.asarray(value, dtype=dtype)

    def require(condition, message):
        if not isinstance(condition, jax.core.Tracer) and not bool(jnp.all(condition)):
            raise ValueError(message)

    for name, value in (("delta_t", delta_t), ("f_lower", f_lower)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if f_lower >= 0.5 / delta_t:
        raise ValueError("f_lower must be below the Nyquist frequency")
    if not math.isfinite(t_pad) or t_pad < 0:
        raise ValueError("t_pad must be finite and nonnegative")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    if f_final is not None:
        raise ValueError("IMRPhenomT does not support the frequency-domain f_final cutoff")
    f_ref = kwargs.pop("f_ref", None)
    if f_ref is None:
        f_ref = f_lower
    else:
        f_ref = array(f_ref)
        f_ref = jnp.where(f_ref == 0, array(f_lower), f_ref)
    require(jnp.isfinite(array(f_ref)) & (array(f_ref) > 0), "f_ref must be finite and nonnegative")
    if kwargs:
        raise TypeError(f"Unsupported IMRPhenomT options: {', '.join(sorted(kwargs))}")

    def spin(chi, z, label):
        if chi is not None and z is not None:
            require(jnp.abs(array(chi) - array(z)) <= 1e-12, f"Conflicting {label} definitions")
        result = array(z if z is not None else chi if chi is not None else 0.0)
        require(jnp.isfinite(result) & (jnp.abs(result) <= 1), "Aligned spins must lie in [-1, 1]")
        return result

    s1, s2 = spin(chi1, spin1z, "spin1"), spin(chi2, spin2z, "spin2")
    for value in (spin1x, spin1y, spin2x, spin2y):
        require(array(value) == 0, "IMRPhenomT only supports aligned spins")
    for value in (lambda1, lambda2):
        if value is not None:
            require(array(value) == 0, "IMRPhenomT does not support tidal deformability")
    m1, m2, dist = array(mass1), array(mass2), array(distance)
    for name, value in (("mass1", m1), ("mass2", m2), ("distance", dist)):
        require(jnp.isfinite(value) & (value > 0), f"{name} must be finite and positive")
    for name, value in (("inclination", inclination), ("phic", phic), ("tc", tc)):
        require(jnp.isfinite(array(value)), f"{name} must be finite")

    model = IMRPhenomT()
    start = model.time_at_frequency(array(f_lower), m1, m2, chi1=s1, chi2=s2)
    require(jnp.isfinite(start), "f_lower is outside the IMRPhenomT frequency range")
    if duration is None:
        # Grid shape is necessarily static; automatic sizing is an eager operation.
        duration = max(2.0, float(jnp.max(-start - array(tc))) + t_pad + delta_t)
    n = int(math.ceil(duration / delta_t))
    actual_duration = n * delta_t
    times = jnp.arange(n, dtype=dtype) * delta_t
    if center_coalescence:
        times = times - (actual_duration - t_pad)
    hp, hc = model.time_domain(
        times,
        m1,
        m2,
        chi1=s1,
        chi2=s2,
        distance=dist,
        inclination=array(inclination),
        phic=array(phic),
        tc=array(tc),
        f_ref=array(f_ref),
    )
    active = times >= (start + array(tc))[..., None]
    hp = jnp.where(active, hp, 0.0)
    hc = jnp.where(active, hc, 0.0)
    return (hp, hc, times) if return_times else (hp, hc)
