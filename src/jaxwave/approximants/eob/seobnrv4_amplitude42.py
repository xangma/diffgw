"""Published pre-NQC (4,2) even-parity modulus in JAX.

Damour--Iyer--Nagar 0811.2069v2 Eq. (102); Pan et al. 1006.0431v2
Eqs. (29h), (38h), (42h). This declared hybrid does not establish
the complete historical SEOBNRv4 flux.
"""

import math

import jax
import jax.numpy as jnp

from diffgw_core.constants import EULER_GAMMA

from .seobnrv4_amplitudes33_44 import _tail_modulus


def seobnrv4_published_amplitude42(
    omega, *, omega_zero_radial, heff_over_mu, hreal_over_M,
    x1, x2, chi1=0., chi2=0.,
):
    """Return ``(R/M)|h42|`` before NQC with the even-parity H source.

    Invalid concrete inputs raise; traced invalid inputs return NaN.
    """
    raw_omega = jnp.asarray(omega)
    if raw_omega.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("omega must have float32 or float64 dtype")
    values = {
        "omega": raw_omega,
        "omega_zero_radial": jnp.asarray(omega_zero_radial, dtype=raw_omega.dtype),
        "heff_over_mu": jnp.asarray(heff_over_mu, dtype=raw_omega.dtype),
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
    for key in ("omega", "omega_zero_radial", "heff_over_mu", "hreal_over_M"):
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
    el2 = EULER_GAMMA+math.log(4)+jnp.log(v)
    rho = (1+(285*nu**2-3530*nu+1146)/(1320*(3*nu-1))*v**2
           -((78*nu**2-59*nu+10)*cs+(10-21*nu)*delta*ca)
           /(15*(1-3*nu))*v**3
           +((-379526805*nu**4-3047981160*nu**3+1204388696*nu**2
              +295834536*nu-114859044)/(317116800*(1-3*nu)**2)
             +q**2/2)*v**4
           -7*q/110*v**5
           +(848238724511/219761942400-3142*el2/3465
             +2323*q**2/3960)*v**6)
    amplitude = (8/63*math.sqrt(math.pi)*nu*vphi**4*values["heff_over_mu"]
                 *_tail_modulus(4, 2, w, values["hreal_over_M"])
                 *(1-3*nu)*jnp.abs(rho)**4)
    return jnp.where(valid, amplitude, jnp.nan)


__all__ = ["seobnrv4_published_amplitude42"]
