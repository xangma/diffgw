"""Sampling and validation for the native IMRPhenomT waveform family."""

import math
import jax
import jax.numpy as jnp

from .approximants.imrphenomt import IMRPhenomT
from diffgw_core.phenomt_options import resolve_precession


def generate_td(
    *,
    approximant="imrphenomt",
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
    names = {
        "imrphenomt": "IMRPhenomT", "imrphenomthm": "IMRPhenomTHM",
        "imrphenomtp": "IMRPhenomTP", "imrphenomtphm": "IMRPhenomTPHM",
    }
    model_name = names[approximant]
    precessing = approximant in ("imrphenomtp", "imrphenomtphm")
    model_options = {}
    if approximant != "imrphenomt" and "modes" in kwargs:
        model_options["modes"] = kwargs.pop("modes")
    if precessing:
        method, version = resolve_precession(
            kwargs.pop("precession_method", "nnlo"), kwargs.pop("precession_version", None),
        )
        model_options.update(precession_method=method, precession_version=version)

    def array(value):
        return jnp.asarray(value, dtype=dtype)

    def require(condition, message):
        if not isinstance(condition, jax.core.Tracer) and not bool(jnp.all(condition)):
            raise ValueError(message)

    for name, value in (("delta_t", delta_t), ("f_lower", f_lower)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if precessing and method == "numerical" and f_lower < 1.0:
        raise ValueError("numerical precession requires f_lower >= 1 Hz")
    if f_lower >= 0.5 / delta_t:
        raise ValueError("f_lower must be below the Nyquist frequency")
    if not math.isfinite(t_pad) or t_pad < 0:
        raise ValueError("t_pad must be finite and nonnegative")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    if f_final is not None:
        raise ValueError(f"{model_name} does not support the frequency-domain f_final cutoff")
    f_ref = kwargs.pop("f_ref", None)
    if f_ref is None:
        f_ref = f_lower
    else:
        f_ref = array(f_ref)
        f_ref = jnp.where(f_ref == 0, array(f_lower), f_ref)
    require(jnp.isfinite(array(f_ref)) & (array(f_ref) > 0), "f_ref must be finite and nonnegative")
    if kwargs:
        raise TypeError(f"Unsupported {model_name} options: {', '.join(sorted(kwargs))}")

    def spin(chi, z, label):
        if chi is not None and z is not None:
            require(jnp.abs(array(chi) - array(z)) <= 1e-12, f"Conflicting {label} definitions")
        result = array(z if z is not None else chi if chi is not None else 0.0)
        require(jnp.isfinite(result) & (jnp.abs(result) <= 1), "Aligned spins must lie in [-1, 1]")
        return result

    s1, s2 = spin(chi1, spin1z, "spin1"), spin(chi2, spin2z, "spin2")
    if precessing:
        for sx, sy, sz in ((spin1x, spin1y, s1), (spin2x, spin2y, s2)):
            norm_sq = array(sx)**2 + array(sy)**2 + sz**2
            require(jnp.isfinite(norm_sq) & (norm_sq <= 1), "Spin-vector norms must not exceed 1")
        model_options.update(spin1x=array(spin1x), spin1y=array(spin1y),
                             spin2x=array(spin2x), spin2y=array(spin2y))
    else:
        for value in (spin1x, spin1y, spin2x, spin2y):
            require(array(value) == 0, f"{model_name} only supports aligned spins")
    for value in (lambda1, lambda2):
        if value is not None:
            require(array(value) == 0, f"{model_name} does not support tidal deformability")
    m1, m2, dist = array(mass1), array(mass2), array(distance)
    for name, value in (("mass1", m1), ("mass2", m2), ("distance", dist)):
        require(jnp.isfinite(value) & (value > 0), f"{name} must be finite and positive")
    for name, value in (("inclination", inclination), ("phic", phic), ("tc", tc)):
        require(jnp.isfinite(array(value)), f"{name} must be finite")

    if approximant == "imrphenomthm":
        from .approximants.imrphenomthm import IMRPhenomTHM
        model = IMRPhenomTHM()
    elif precessing:
        from .approximants.imrphenomtp import IMRPhenomTP
        from .approximants.imrphenomtphm import IMRPhenomTPHM
        model = IMRPhenomTP() if approximant == "imrphenomtp" else IMRPhenomTPHM()
    else:
        model = IMRPhenomT()
    final_spin = None
    if precessing:
        from .approximants._phenomt_dispatch import prepare_precession
        final_spin, prepared = prepare_precession(
            method, m1, m2, array(spin1x), array(spin1y), s1,
            array(spin2x), array(spin2y), s2, array(f_ref),
        )
        model_options.update(final_spin=final_spin, _precession_state=prepared)
    start = IMRPhenomT().time_at_frequency(
        array(f_lower), m1, m2, chi1=s1, chi2=s2,
        _final_spin_override=final_spin,
    )
    require(jnp.isfinite(start), f"f_lower is outside the {model_name} frequency range")
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
        **model_options,
    )
    active = times >= (start + array(tc))[..., None]
    hp = jnp.where(active, hp, 0.0)
    hc = jnp.where(active, hc, 0.0)
    return (hp, hc, times) if return_times else (hp, hc)
