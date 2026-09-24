"""Experimental paper-based SEOBNRv4 pre-NQC (2,2) phase and mode.

The spin-phase inheritance switches are required because the published
equations do not select a unique historical-v4 prescription. This is a
component, not a qualified trajectory or complete waveform.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .seobnrv4_amplitude22 import seobnrv4_published_amplitude22


def _inputs(omega, **kwargs):
    raw = jnp.asarray(omega)
    if raw.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("omega must have float32 or float64 dtype")
    for key, value in kwargs.items():
        if jnp.asarray(value).dtype.kind == "c":
            raise ValueError(f"{key} must be real")
    keys = ("omega", *kwargs)
    values = jnp.broadcast_arrays(raw, *(jnp.asarray(value, dtype=raw.dtype)
                                         for value in kwargs.values()))
    result = dict(zip(keys, values))
    valid = jnp.ones_like(result["omega"], dtype=bool)
    for value in result.values():
        valid &= jnp.isfinite(value)
    for key in ("omega", "hreal_over_M"):
        if key in result:
            valid &= result[key] > 0
    if "nu" in result:
        valid &= (result["nu"] >= 0) & (result["nu"] <= .25)
    for key in ("q_kerr", "chi", "chi1", "chi2"):
        if key in result:
            valid &= jnp.abs(result[key]) <= 1
    if "x1" in result and "x2" in result:
        a, b = result["x1"], result["x2"]
        valid &= (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
        valid &= jnp.abs(a+b-1) <= 16*jnp.finfo(raw.dtype).eps
    if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
        raise ValueError("Require finite physical frequencies, energies, mass fractions and spins")
    return result, valid


def _masked(value, valid):
    return jnp.where(valid, value, jnp.nan)


def tail22_phase(omega, *, hreal_over_M):
    """Continuous Taracchini (A6) tail phase, with r0/M=2/sqrt(e)."""
    p, valid = _inputs(omega, hreal_over_M=hreal_over_M)
    frequency, energy = p["omega"], p["hreal_over_M"]
    khat = 2*frequency*energy
    z = 3-2j*khat
    w = z+13
    log_gamma = (w-.5)*jnp.log(w)-w+.5*math.log(2*math.pi)
    bernoulli = (1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510)
    for k, number in enumerate(bernoulli, 1):
        log_gamma += number/(2*k*(2*k-1))*w**(-(2*k-1))
    for s in range(13):
        log_gamma -= jnp.log(z+s)
    # The recurrence uses the DIN radius 2M; Taracchini uses 2M/sqrt(e).
    return _masked(log_gamma.imag+2*khat*jnp.log(8*frequency)-khat, valid)


def seobnrv4_published_delta22(
    omega, *, hreal_over_M, x1, x2, chi1, chi2,
    spin_v6, include_pan_v8, include_positive_chi_540,
):
    """Compose explicit printed residual-phase choices, without a default."""
    if spin_v6 not in ("bohe_total_energy", "pan_test_particle", "bohe_plus_pan", "none"):
        raise ValueError("spin_v6 must explicitly select a supported source choice")
    if type(include_pan_v8) is not bool or type(include_positive_chi_540) is not bool:
        raise ValueError("Pan v8 and positive-chi calibration switches must be explicit bools")
    p, valid = _inputs(omega, hreal_over_M=hreal_over_M, x1=x1, x2=x2,
                       chi1=chi1, chi2=chi2)
    w, e, a, b, s1, s2 = (p[key] for key in
                          ("omega", "hreal_over_M", "x1", "x2", "chi1", "chi2"))
    nu = a*b
    q_kerr = (1-2*nu)*(s1+s2)/2+(a-b)*(s1-s2)/2
    u = w*e
    result = (7*u/3+428*math.pi*u**2/105
              +(1712*math.pi**2/315-2203/81)*u**3-24*nu*w**(5/3))
    if spin_v6 in ("bohe_total_energy", "bohe_plus_pan"):
        result -= 4*q_kerr*u**2/3
    if spin_v6 in ("pan_test_particle", "bohe_plus_pan"):
        result -= 4*q_kerr*w**2/3
    if include_pan_v8:
        result += 20*q_kerr*w**(8/3)/63
    if include_positive_chi_540:
        chi = q_kerr/(1-2*nu)
        result += 540*nu*jnp.where(chi >= 0, chi, 0.)*w**2
    return _masked(result, valid)


def seobnrv4_published_phase22(omega, phi, *, hreal_over_M, residual_phase):
    """Continuous pre-NQC GW phase, including the Newtonian minus sign."""
    p, valid = _inputs(omega, phi=phi, hreal_over_M=hreal_over_M,
                       residual_phase=residual_phase)
    phase = 2*p["phi"]-tail22_phase(p["omega"], hreal_over_M=p["hreal_over_M"])
    return _masked(phase-p["residual_phase"]+math.pi, valid)


def seobnrv4_published_mode22(
    omega, phi, *, residual_phase, omega_zero_radial, heff_over_mu,
    hreal_over_M, x1, x2, chi1=0., chi2=0., retain_pan_cubic_spin=False,
):
    """Complex pre-NQC (R/M)h22 for an explicitly supplied residual phase."""
    phase = seobnrv4_published_phase22(
        omega, phi, hreal_over_M=hreal_over_M, residual_phase=residual_phase)
    amplitude = seobnrv4_published_amplitude22(
        omega, omega_zero_radial=omega_zero_radial, heff_over_mu=heff_over_mu,
        hreal_over_M=hreal_over_M, x1=x1, x2=x2, chi1=chi1, chi2=chi2,
        retain_pan_cubic_spin=retain_pan_cubic_spin)
    return amplitude*jnp.exp(-1j*phase)


__all__ = ["tail22_phase", "seobnrv4_published_delta22",
           "seobnrv4_published_phase22", "seobnrv4_published_mode22"]
