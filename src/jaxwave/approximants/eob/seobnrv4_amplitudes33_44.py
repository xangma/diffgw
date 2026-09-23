"""Published pre-NQC (3,3) and (4,4) factorized moduli in JAX.

Damour--Iyer--Nagar 0811.2069 Eqs. (52), (62) and Pan et al.
1006.0431v2 Eqs. (28c), (29f), (38c), (38f), (41c), (42f).
These paper terms do not specify the complete historical SEOBNRv4 flux.
"""

import math

import jax
import jax.numpy as jnp

from diffgw_core.constants import EULER_GAMMA


def _tail_modulus(ell, m, omega, hreal_over_M):
    khat = m*omega*hreal_over_M
    squared = 4*math.pi*khat/(-jnp.expm1(-4*math.pi*khat))
    for degree in range(1, ell+1):
        squared *= 1+4*khat**2/degree**2
    return jnp.sqrt(squared)


def seobnrv4_published_amplitudes33_44(
    omega, *, omega_zero_radial, heff_over_mu, hreal_over_M,
    x1, x2, chi1=0., chi2=0.,
):
    """Return ``((R/M)|h33|, (R/M)|h44|)`` before NQC or merger.

    ``omega_zero_radial`` is evaluated at the same radius and angular
    momentum as ``omega``, setting only radial momentum to zero. Energies
    include the normalization described for the published (2,2) amplitude.
    Invalid concrete inputs raise; traced invalid inputs return NaN.
    """
    raw_omega = jnp.asarray(omega)
    if raw_omega.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("omega must have float32 or float64 dtype")
    inputs = {
        "omega": raw_omega,
        "omega_zero_radial": jnp.asarray(omega_zero_radial, dtype=raw_omega.dtype),
        "heff_over_mu": jnp.asarray(heff_over_mu, dtype=raw_omega.dtype),
        "hreal_over_M": jnp.asarray(hreal_over_M, dtype=raw_omega.dtype),
        "x1": jnp.asarray(x1, dtype=raw_omega.dtype),
        "x2": jnp.asarray(x2, dtype=raw_omega.dtype),
        "chi1": jnp.asarray(chi1, dtype=raw_omega.dtype),
        "chi2": jnp.asarray(chi2, dtype=raw_omega.dtype),
    }
    inputs = dict(zip(inputs, jnp.broadcast_arrays(*inputs.values())))
    valid = jnp.ones_like(inputs["omega"], dtype=bool)
    for value in inputs.values():
        valid &= jnp.isfinite(value)
    for key in ("omega", "omega_zero_radial", "heff_over_mu", "hreal_over_M"):
        valid &= inputs[key] > 0
    a, b = inputs["x1"], inputs["x2"]
    valid &= (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
    valid &= jnp.abs(a+b-1) <= 16*jnp.finfo(raw_omega.dtype).eps
    valid &= (jnp.abs(inputs["chi1"]) <= 1) & (jnp.abs(inputs["chi2"]) <= 1)
    if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
        raise ValueError("Require finite positive frequencies/energies, physical mass fractions and spins")

    nu, delta = a*b, a-b
    cs = (inputs["chi1"]+inputs["chi2"])/2
    ca = (inputs["chi1"]-inputs["chi2"])/2
    q = (1-2*nu)*cs+delta*ca
    w, w0 = inputs["omega"], inputs["omega_zero_radial"]
    v, vphi = jnp.cbrt(w), w*w0**(-2/3)
    el3 = EULER_GAMMA+math.log(6)+jnp.log(v)
    el4 = EULER_GAMMA+math.log(8)+jnp.log(v)

    rho33 = (1+(2*nu/3-7/6)*v**2
             +(149*nu**2/330-1861*nu/990-6719/3960)*v**4
             +(3203101567/227026800-26*el3/7)*v**6
             +(13*el3/3-57566572157/8562153600)*v**8)
    higher33 = (3*q**2/2*v**4+2*q/3*v**5-7*q**2/4*v**6
                +(6187*q/330-q**3)*v**7)
    residual33 = (delta*(rho33**3+higher33)
                  -((2-5*nu/2)*delta*cs+(2-19*nu/2)*ca)*v**3)

    rho44 = (1+(2625*nu**2-5870*nu+1614)/(1320*(3*nu-1))*v**2
             +(1252563795*nu**4-6733146000*nu**3-313857376*nu**2
               +2338945704*nu-511573572)/(317116800*(1-3*nu)**2)*v**4
             +(16600939332793/1098809712000-12568*el4/3465)*v**6
             -((42*nu**2-41*nu+10)*cs+(10-39*nu)*delta*ca)
             /(15*(1-3*nu))*v**3
             +q**2/2*v**4-69*q/55*v**5+217*q**2/3960*v**6)

    heff, hreal = inputs["heff_over_mu"], inputs["hreal_over_M"]
    amp33 = 3*math.sqrt(6*math.pi/7)*nu*vphi**3*heff*_tail_modulus(3, 3, w, hreal)*jnp.abs(residual33)
    amp44 = (64/9*math.sqrt(math.pi/7)*nu*vphi**4*heff
             *_tail_modulus(4, 4, w, hreal)*(1-3*nu)*jnp.abs(rho44)**4)
    return jnp.where(valid, amp33, jnp.nan), jnp.where(valid, amp44, jnp.nan)


__all__ = ["seobnrv4_published_amplitudes33_44"]
