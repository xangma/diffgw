"""Published SEOBNRv4 pre-NQC (2,2) amplitude in JAX.

The residual follows Taracchini et al. 1202.0790, Appendix A; Pan et al.
1006.0431v2, Eqs. (29a), (42a); Buonanno et al. 1209.6349v2,
Eq. (4.27b); and Bohé et al. 1611.03703v3, Eqs. (2.4)--(2.5).
This is a paper reconstruction, not a complete historical-v4 waveform.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from diffgw_core.constants import EULER_GAMMA


def _rho22(v, nu, delta, chi_s, chi_a, *, retain_pan_cubic_spin=False):
    """Residual rho22 with the Bohé replacement of the 3.5PN spin term."""
    if not isinstance(retain_pan_cubic_spin, bool):
        raise TypeError("retain_pan_cubic_spin must be an explicit bool")
    kerr = (1 - 2*nu)*chi_s + delta*chi_a
    chi_zero = chi_s + delta*chi_a
    eulerlog = EULER_GAMMA + math.log(4) + jnp.log(v)
    c2 = 55*nu/84 - 43/42
    c3 = -2*((1-nu)*chi_s + delta*chi_a)/3
    c4 = 19583*nu**2/42336 - 33025*nu/21168 - 20555/10584 + chi_zero**2/2
    c5 = (-34/21 + 49*nu/18 + 209*nu**2/126)*chi_s - (34/21 + 19*nu/42)*delta*chi_a
    c6 = (10620745*nu**3/39118464 - 6292061*nu**2/3259872
          + 41*math.pi**2*nu/192 - 48993925*nu/9779616
          - 428*eulerlog/105 + 1556919113/122245200 + 89*kerr**2/252)
    c7 = ((18733/15876 + 74749*nu/5292 - 245717*nu**2/63504
           + 50803*nu**3/63504)*chi_s
          + (18733/15876 + 50140*nu/3969 + 97865*nu**2/63504)*delta*chi_a)
    if retain_pan_cubic_spin:
        c7 += kerr**3/3
    c8 = 9202*eulerlog/2205 - 387216563023/160190110080 + 18353*kerr**2/21168 - kerr**4/8
    c10 = 439877*eulerlog/55566 - 16094530514677/533967033600
    return (1 + c2*v**2 + c3*v**3 + c4*v**4 + c5*v**5
            + c6*v**6 + c7*v**7 + c8*v**8 + c10*v**10)


def seobnrv4_published_amplitude22(
    omega,
    *,
    omega_zero_radial,
    heff_over_mu,
    hreal_over_M,
    x1,
    x2,
    chi1=0.,
    chi2=0.,
    retain_pan_cubic_spin=False,
):
    """Return dimensionless ``(R/M)|h22|`` before NQC or merger attachment.

    Frequencies use ``M*d(phi)/dt``. ``omega_zero_radial`` holds ``r,p_phi``
    fixed while setting radial momentum to zero. Both energies include the
    normalization stated in the Torch counterpart; ``hreal_over_M`` includes
    rest mass. Inputs broadcast and retain the frequency's float32/float64
    dtype. Invalid concrete inputs raise; traced invalid inputs yield NaN.
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
        raise ValueError("Require finite positive frequencies/energies, physical mass fractions and spins")

    nu = a*b
    chi_s = (values["chi1"]+values["chi2"])/2
    chi_a = (values["chi1"]-values["chi2"])/2
    v = jnp.cbrt(values["omega"])
    v_phi = values["omega"]*values["omega_zero_radial"]**(-2/3)
    rho = _rho22(v, nu, a-b, chi_s, chi_a,
                 retain_pan_cubic_spin=retain_pan_cubic_spin)
    khat = 2*values["omega"]*values["hreal_over_M"]
    tail_squared = (4*math.pi*khat/(-jnp.expm1(-4*math.pi*khat))
                    * (1+4*khat**2)*(1+khat**2))
    amplitude = 8*math.sqrt(math.pi/5)*nu*v_phi**2*values["heff_over_mu"]*jnp.sqrt(tail_squared)*rho**2
    return jnp.where(valid, amplitude, jnp.nan)


__all__ = ["seobnrv4_published_amplitude22"]
