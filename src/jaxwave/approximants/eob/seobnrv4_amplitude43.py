"""Conditional pre-NQC (4,3) J-source modulus in JAX.

Damour--Iyer--Nagar 0811.2069v2 Eq. (101); Pan et al. 1006.0431v2
Eqs. (41g), (42g); Taracchini et al. 1202.0790v2 Eq. (52b).
The regularized paper hybrid does not fix every historical mode choice.
"""

import math

import jax
import jax.numpy as jnp

from diffgw_core.constants import EULER_GAMMA

from .seobnrv4_amplitudes33_44 import _tail_modulus


def seobnrv4_published_amplitude43(
    omega, *, omega_zero_radial, pphi, hreal_over_M,
    x1, x2, chi1=0., chi2=0.,
):
    """Return ``(R/M)|h43|`` before NQC with ``pphi=|L|/(mu*M)``.

    The odd-mass residual is formed without dividing by the mass difference.
    Invalid concrete inputs raise; traced invalid inputs return NaN.
    """
    raw_omega = jnp.asarray(omega)
    if raw_omega.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("omega must have float32 or float64 dtype")
    values = {
        "omega": raw_omega,
        "omega_zero_radial": jnp.asarray(omega_zero_radial, dtype=raw_omega.dtype),
        "pphi": jnp.asarray(pphi, dtype=raw_omega.dtype),
        "hreal_over_M": jnp.asarray(hreal_over_M, dtype=raw_omega.dtype),
        "x1": jnp.asarray(x1, dtype=raw_omega.dtype),
        "x2": jnp.asarray(x2, dtype=raw_omega.dtype),
        "chi1": jnp.asarray(chi1, dtype=raw_omega.dtype),
        "chi2": jnp.asarray(chi2, dtype=raw_omega.dtype),
    }
    values = dict(zip(values, jnp.broadcast_arrays(*values.values())))
    valid = jnp.ones_like(values["omega"], dtype=bool)
    for value in values.values():
        valid &= jnp.isfinite(value)
    for key in ("omega", "omega_zero_radial", "pphi", "hreal_over_M"):
        valid &= values[key] > 0
    a, b = values["x1"], values["x2"]
    valid &= (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
    valid &= jnp.abs(a+b-1) <= 16*jnp.finfo(raw_omega.dtype).eps
    valid &= (jnp.abs(values["chi1"]) <= 1) & (jnp.abs(values["chi2"]) <= 1)
    if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
        raise ValueError("Require finite positive orbital inputs, physical mass fractions and spins")

    nu, delta = a*b, a-b
    cs = (values["chi1"]+values["chi2"])/2
    ca = (values["chi1"]-values["chi2"])/2
    q = (1-2*nu)*cs+delta*ca
    w, w0 = values["omega"], values["omega_zero_radial"]
    v, vphi = jnp.cbrt(w), w*w0**(-2/3)
    el3 = EULER_GAMMA+math.log(6)+jnp.log(v)
    rho = (1+(160*nu**2-547*nu+222)/(176*(2*nu-1))*v**2
           -6894273/7047040*v**4
           +(1664224207351/195343948800-1571*el3/770)*v**6)
    residual = ((b**4-a**4)*(rho**4+3*q**2/2*v**4-12113*q/1540*v**5)
                -5*nu/2*(delta*cs-ca)*v)
    amplitude = (9/5*math.sqrt(2*math.pi/7)*nu*vphi**5*values["pphi"]*v
                 *_tail_modulus(4, 3, w, values["hreal_over_M"])
                 *jnp.abs(residual))
    return jnp.where(valid, amplitude, jnp.nan)


__all__ = ["seobnrv4_published_amplitude43"]
