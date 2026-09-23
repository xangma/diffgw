"""Published pre-NQC (3,2) J-source modulus in JAX.

Damour--Iyer--Nagar 0811.2069 Eq. (53); Pan et al. 1006.0431v2
Eqs. (29d), (38d), (42d), Appendix F.1. This declared paper hybrid
does not establish the complete historical SEOBNRv4 flux.
"""

import math

import jax
import jax.numpy as jnp

from diffgw_core.constants import EULER_GAMMA

from .seobnrv4_amplitudes33_44 import _tail_modulus


def seobnrv4_published_amplitude32(
    omega, *, omega_zero_radial, pphi, hreal_over_M,
    x1, x2, chi1=0., chi2=0.,
):
    """Return ``(R/M)|h32|`` before NQC with ``pphi=|L|/(mu*M)``.

    The zero-radial frequency holds radius and angular momentum fixed.
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
    el2 = EULER_GAMMA+math.log(4)+jnp.log(v)
    spin_linear = -4*nu*cs/(3*(3*nu-1))
    rho = (1+spin_linear*v
           +(320*nu**2-1115*nu+328)/(270*(3*nu-1))*v**2
           -spin_linear**2*v**2+2*q/9*v**3
           +(3085640*nu**4-20338960*nu**3-4725605*nu**2
             +8050045*nu-1444528)/(1603800*(1-3*nu)**2)*v**4
           +q**2/3*v**4-2788*q/1215*v**5
           +(5849948554/940355325-104*el2/63+488*q**2/405)*v**6
           +(17056*el2/8505-10607269449358/3072140846775)*v**8)
    amplitude = (8/3*math.sqrt(math.pi/7)*nu*vphi**4*values["pphi"]*v
                 *_tail_modulus(3, 2, w, values["hreal_over_M"])
                 *(1-3*nu)*jnp.abs(rho)**3)
    return jnp.where(valid, amplitude, jnp.nan)


__all__ = ["seobnrv4_published_amplitude32"]
