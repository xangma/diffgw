"""Paper-derived pre-NQC SEOBNRv4 factorized modes and energy flux.

DIN 0811.2069 Appendix C; Pan 1006.0431 Eqs. (28), (29), (33), (41),
(42), (F26), Appendix D; Taracchini 1202.0790 Appendix A and 1311.2544;
BFH 1209.6349 Eq. (4.27); Bohé 1611.03703 Eq. (2.9).
"""

from collections.abc import Mapping
import math

from diffgw_core.constants import EULER_GAMMA
from .seobnrv4_amplitude22 import _rho22
from .seobnrv4_amplitudes33_44 import _tail_modulus

import jax
import jax.numpy as jnp


def emitted_flux_from_modes(omega, positive_m_modes: Mapping[tuple[int, int], object]):
    """Return ``sum((m*omega)**2 * |R*h_lm/M|**2)/(8*pi)``.

    Inputs broadcast, and ``omega`` is the dimensionless orbital frequency.
    Concrete invalid inputs raise; compiled invalid rows return NaN. Supply
    only pre-NQC, positive-m modes with ``2<=ell<=8``. A partial mapping gives
    a partial flux, not the full SEOBNRv4 torque.
    """
    if not isinstance(positive_m_modes, Mapping) or not positive_m_modes:
        raise ValueError("at least one positive-m mode is required")
    for mode in positive_m_modes:
        if (not isinstance(mode, tuple) or len(mode) != 2
                or any(type(index) is not int for index in mode)
                or not 2 <= mode[0] <= 8 or not 1 <= mode[1] <= mode[0]):
            raise ValueError("mode keys must satisfy 2<=ell<=8 and 1<=m<=ell")

    frequency = jnp.asarray(omega)
    if frequency.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("omega must have float32 or float64 dtype")
    amplitudes = {
        mode: jnp.asarray(value, dtype=jnp.result_type(frequency.dtype, value))
        for mode, value in positive_m_modes.items()
    }
    shapes = jnp.broadcast_arrays(frequency, *amplitudes.values())
    frequency = shapes[0]
    amplitudes = dict(zip(amplitudes, shapes[1:]))
    valid = jnp.isfinite(frequency) & (frequency > 0)
    total = jnp.zeros_like(frequency)
    for (_, m), amplitude in amplitudes.items():
        valid &= jnp.isfinite(amplitude)
        total = total + (m*frequency)**2 * (jnp.real(amplitude)**2 + jnp.imag(amplitude)**2)
    if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
        raise ValueError("frequency must be positive and all inputs finite")
    return jnp.where(valid, total/(8*math.pi), jnp.nan)




# Paper-derived DIN, Pan, Taracchini, BFH and Bohe residuals.
# These pure-JAX formulas mirror the repository reconstruction, not a ROM table.
def _double_factorial(n: int) -> int:
    return math.prod(range(n, 0, -2))


def _newtonian_constant(ell: int, m: int) -> complex:
    """DIN (5)--(6), times Y_(ell-epsilon,-m)(pi/2,0)."""
    epsilon = (ell + m) % 2
    if epsilon == 0:
        n = (1j*m)**ell * 8*math.pi / _double_factorial(2*ell+1)
        n *= math.sqrt((ell+1)*(ell+2)/(ell*(ell-1)))
    else:
        n = -(1j*m)**ell * 16*math.pi*1j / _double_factorial(2*ell+1)
        n *= math.sqrt((2*ell+1)*(ell+2)*(ell*ell-m*m)
                       / ((2*ell-1)*(ell+1)*ell*(ell-1)))
    degree = ell-epsilon
    p_lm = (-1)**((degree+m)//2)*_double_factorial(degree+m-1)
    p_lm /= _double_factorial(degree-m)
    y_l_minus_m = (-1)**m*p_lm*math.sqrt(
        (2*degree+1)/(4*math.pi)*math.factorial(degree-m)/math.factorial(degree+m)
    )
    return n*y_l_minus_m


def _rho_table(x: jax.Array, nu: jax.Array) -> dict[tuple[int, int], jax.Array]:
    """All 35 hybridized J-source residual amplitudes, DIN (C1)--(C35).

    Terms beyond the displayed order of each equation are omitted. In
    particular, higher test-mass coefficients are exactly those published
    in Appendix C; they are not promoted to finite-mass PN information.
    """
    n = nu
    el = {m: EULER_GAMMA + math.log(2*m) + .5*jnp.log(x) for m in range(1, 5)}
    rho = {
        # C1--C5
        (2, 2): 1 + (55*n/84-43/42)*x
        + (19583*n**2/42336-33025*n/21168-20555/10584)*x**2
        + (10620745*n**3/39118464-6292061*n**2/3259872+41*math.pi**2*n/192
           -48993925*n/9779616-428*el[2]/105+1556919113/122245200)*x**3
        + (9202*el[2]/2205-387216563023/160190110080)*x**4
        + (439877*el[2]/55566-16094530514677/533967033600)*x**5,
        (2, 1): 1 + (23*n/84-59/56)*x
        + (617*n**2/4704-10993*n/14112-47009/56448)*x**2
        + (7613184941/2607897600-107*el[1]/105)*x**3
        + (6313*el[1]/5880-1168617463883/911303737344)*x**4,
        (3, 3): 1 + (2*n/3-7/6)*x
        + (149*n**2/330-1861*n/990-6719/3960)*x**2
        + (3203101567/227026800-26*el[3]/7)*x**3
        + (13*el[3]/3-57566572157/8562153600)*x**4,
        (3, 2): 1 + (320*n**2-1115*n+328)/(270*(3*n-1))*x
        + (3085640*n**4-20338960*n**3-4725605*n**2+8050045*n-1444528)
        / (1603800*(1-3*n)**2)*x**2
        + (5849948554/940355325-104*el[2]/63)*x**3,
        (3, 1): 1 + (-2*n/9-13/18)*x
        + (-829*n**2/1782-1685*n/1782+101/7128)*x**2
        + (11706720301/6129723600-26*el[1]/63)*x**3
        + (169*el[1]/567+2606097992581/4854741091200)*x**4,
        # C6--C10
        (4, 4): 1 + (2625*n**2-5870*n+1614)/(1320*(3*n-1))*x
        + (1252563795*n**4-6733146000*n**3-313857376*n**2+2338945704*n-511573572)
        / (317116800*(1-3*n)**2)*x**2
        + (16600939332793/1098809712000-12568*el[4]/3465)*x**3,
        (4, 3): 1 + (160*n**2-547*n+222)/(176*(2*n-1))*x-6894273/7047040*x**2,
        (4, 2): 1 + (285*n**2-3530*n+1146)/(1320*(3*n-1))*x
        + (-379526805*n**4-3047981160*n**3+1204388696*n**2+295834536*n-114859044)
        / (317116800*(1-3*n)**2)*x**2
        + (848238724511/219761942400-3142*el[2]/3465)*x**3,
        (4, 1): 1 + (288*n**2-1385*n+602)/(528*(2*n-1))*x-7775491/21141120*x**2,
        (5, 5): 1 + (512*n**2-1298*n+487)/(390*(2*n-1))*x-3353747/2129400*x**2,
        # C11--C20
        (5, 4): 1 + (33320*n**3-127610*n**2+96019*n-17448)
        / (13650*(5*n**2-5*n+1))*x,
        (5, 3): 1 + (176*n**2-850*n+375)/(390*(2*n-1))*x-410833/709800*x**2,
        (5, 2): 1 + (21980*n**3-104930*n**2+84679*n-15828)
        / (13650*(5*n**2-5*n+1))*x,
        (5, 1): 1 + (8*n**2-626*n+319)/(390*(2*n-1))*x-31877/304200*x**2,
        (6, 6): 1 + (273*n**3-861*n**2+602*n-106)/(84*(5*n**2-5*n+1))*x,
        (6, 5): 1 + (220*n**3-910*n**2+838*n-185)/(144*(3*n**2-4*n+1))*x,
        (6, 4): 1 + (133*n**3-581*n**2+462*n-86)/(84*(5*n**2-5*n+1))*x,
        (6, 3): 1 + (156*n**3-750*n**2+742*n-169)/(144*(3*n**2-4*n+1))*x,
        (6, 2): 1 + (49*n**3-413*n**2+378*n-74)/(84*(5*n**2-5*n+1))*x,
        (6, 1): 1 + (124*n**3-670*n**2+694*n-161)/(144*(3*n**2-4*n+1))*x,
        # C21--C27
        (7, 7): 1 + (1380*n**3-4963*n**2+4246*n-906)/(714*(3*n**2-4*n+1))*x,
        (7, 6): 1 + (6104*n**4-29351*n**3+37828*n**2-16185*n+2144)
        / (1666*(7*n**3-14*n**2+7*n-1))*x,
        (7, 5): 1 + (804*n**3-3523*n**2+3382*n-762)/(714*(3*n**2-4*n+1))*x,
        (7, 4): 1 + (41076*n**4-217959*n**3+298872*n**2-131805*n+17756)
        / (14994*(7*n**3-14*n**2+7*n-1))*x,
        (7, 3): 1 + (420*n**3-2563*n**2+2806*n-666)/(714*(3*n**2-4*n+1))*x,
        (7, 2): 1 + (32760*n**4-190239*n**3+273924*n**2-123489*n+16832)
        / (14994*(7*n**3-14*n**2+7*n-1))*x,
        (7, 1): 1 + (228*n**3-2083*n**2+2518*n-618)/(714*(3*n**2-4*n+1))*x,
        # C28--C35
        (8, 8): 1 + (12243*n**4-53445*n**3+64659*n**2-26778*n+3482)
        / (2736*(7*n**3-14*n**2+7*n-1))*x,
        (8, 7): 1 + (38920*n**4-207550*n**3+309498*n**2-154099*n+23478)
        / (18240*(4*n**3-10*n**2+6*n-1))*x,
        (8, 6): 1 + (2653*n**4-13055*n**3+17269*n**2-7498*n+1002)
        / (912*(7*n**3-14*n**2+7*n-1))*x,
        (8, 5): 1 + (6056*n**4-34598*n**3+54642*n**2-28055*n+4350)
        / (3648*(4*n**3-10*n**2+6*n-1))*x,
        (8, 4): 1 + (4899*n**4-28965*n**3+42627*n**2-19434*n+2666)
        / (2736*(7*n**3-14*n**2+7*n-1))*x,
        (8, 3): 1 + (24520*n**4-149950*n**3+249018*n**2-131059*n+20598)
        / (18240*(4*n**3-10*n**2+6*n-1))*x,
        (8, 2): 1 + (3063*n**4-22845*n**3+37119*n**2-17598*n+2462)
        / (2736*(7*n**3-14*n**2+7*n-1))*x,
        (8, 1): 1 + (21640*n**4-138430*n**3+236922*n**2-126451*n+20022)
        / (18240*(4*n**3-10*n**2+6*n-1))*x,
    }
    return rho




def _odd_mass_leading_spin_f(
    ell: int, m: int, v: jax.Array, x1: jax.Array, x2: jax.Array,
    chi1: jax.Array, chi2: jax.Array,
) -> jax.Array:
    """Odd-m c_(ell+epsilon) f_spin from Pan (41), regular at equal mass.

    This adds the published linear-spin f term to rho_nonspin**ell;
    it does not exponentiate a spin-corrected rho or supply higher PN spin.
    """
    nu = x1*x2
    chi_s, chi_a = (chi1+chi2)/2, (chi1-chi2)/2
    p = ell+(ell+m)%2-1
    c = x2**p+(-1)**m*x1**p
    c_over_delta = -sum(x1**j*x2**(p-1-j) for j in range(p))
    if (ell, m) == (2, 1):
        return -1.5*(c*chi_s+c_over_delta*chi_a)*v
    if (ell, m) == (3, 3):
        return -((2-5*nu/2)*c*chi_s+(2-19*nu/2)*c_over_delta*chi_a)*v**3
    if (ell, m) == (3, 1):
        return -((2-13*nu/2)*c*chi_s+(2-11*nu/2)*c_over_delta*chi_a)*v**3
    if ell == 4 and m % 2:
        return -5*nu/(2*(2*nu-1))*(c*chi_s-c_over_delta*chi_a)*v
    raise NotImplementedError("This spin residual is tabulated only for odd m through ell=4")




def _nonspin_rho(v: jax.Array, nu: jax.Array) -> dict[tuple[int, int], jax.Array]:
    """Taracchini (A8)--(A14), dropping calibrated nu*rho22^(4).

    DIN (C1)--(C35) already supplies the common terms. Nine additional
    test-particle terms are printed in Taracchini. Retain DIN/Pan's rho55
    v**2 sign: the extra minus in Taracchini (A11a) fails their nu=0 limit.
    """
    rho = _rho_table(v*v, nu)
    el = {m: EULER_GAMMA+math.log(2*m)+jnp.log(v) for m in (1, 2, 3)}
    extra = {
        (2, 1): (5029963*el[1]/5927040-63735873771463/16569158860800)*v**10,
        (3, 2): (17056*el[2]/8505-10607269449358/3072140846775)*v**8,
        (4, 3): (1664224207351/195343948800-1571*el[3]/770)*v**6,
        (4, 1): (1227423222031/1758095539200-1571*el[1]/6930)*v**6,
        (5, 4): -16213384/15526875*v**4,
        (5, 2): -7187914/15526875*v**4,
        (6, 6): -1025435/659736*v**4,
        (6, 4): -476887/659736*v**4,
        (6, 2): -817991/3298680*v**4,
    }
    for mode, value in extra.items():
        rho[mode] = rho[mode]+value
    return rho


def _spin_rho(v, nu, delta, cs, ca, *, spin_truncation):
    """Spin portions of ordinary rho for modes without odd-m f splitting.

    Kerr spin q=chiK=(1-2nu)chiS+delta*chiA, as specified in Taracchini
    2014. Comparable-mass coefficients replace the same PN spin terms;
    they are never added on top of the test-particle counterparts.
    """
    q = (1-2*nu)*cs+delta*ca
    r32v1 = -4*nu*cs/(3*(3*nu-1))
    result = {
        # The v**2 spin square is Pan (F26), -rho32_v1**2.
        (3, 2): r32v1*v-r32v1**2*v**2+2*q/9*v**3+q*q/3*v**4
        -2788*q/1215*v**5+488*q*q/405*v**6,
        (4, 4): -((42*nu**2-41*nu+10)*cs+(10-39*nu)*delta*ca)
        /(15*(1-3*nu))*v**3+q*q/2*v**4-69*q/55*v**5+217*q*q/3960*v**6,
        (4, 2): -((78*nu**2-59*nu+10)*cs+(10-21*nu)*delta*ca)
        /(15*(1-3*nu))*v**3+q*q/2*v**4-7*q/110*v**5+2323*q*q/3960*v**6,
        (5, 5): -2*q/3*v**3+q*q/2*v**4-241*q/195*v**5,
        (5, 4): -2*q/15*v**3+2*q*q/5*v**4,
        (5, 3): -2*q/3*v**3+q*q/2*v**4-103*q/325*v**5,
        (5, 2): -2*q/15*v**3+2*q*q/5*v**4,
        (5, 1): -2*q/3*v**3+q*q/2*v**4+139*q/975*v**5,
    }
    for m in (2, 4, 6):
        result[6, m] = -2*q/3*v**3+q*q/2*v**4
    for m in (1, 3, 5):
        result[6, m] = -2*q/9*v**3
    for m in (1, 3, 5, 7):
        result[7, m] = -2*q/3*v**3
    if spin_truncation == "generic-leading-ell8":
        for m in (2, 4, 6, 8):
            result[8, m] = -2*q/3*v**3
    return result


def _higher_kerr_f(v: jax.Array, q: jax.Array) -> dict[tuple[int, int], jax.Array]:
    """Pan (28) spin f terms after removing finite-mass replacements.

    Excludes f21's v and v**3 and f33/f31's v**3. f43/f41 have no
    test-particle v term. Only the printed spin terms are imported;
    the entire nonspin factor is supplied separately by rho_NS**ell.
    """
    el1 = EULER_GAMMA+math.log(2)+jnp.log(v)
    return {
        (2, 1): -3*q*q*v**4+3*q*(4*q*q-27)/16*v**5+4163*q*q/252*v**6
        +(-2593*q**3/168+107*q*el1/35-11847887*q/1058400)*v**7,
        (3, 3): 3*q*q/2*v**4+2*q/3*v**5-7*q*q/4*v**6
        +(6187*q/330-q**3)*v**7,
        (3, 1): -5*q*q/2*v**4+38*q/9*v**5+43*q*q/12*v**6
        +(11*q**3/3-2657*q/594)*v**7,
        (4, 3): 3*q*q/2*v**4-12113*q/1540*v**5,
        (4, 1): 3*q*q/2*v**4+(-10*q**3/3-20033*q/13860)*v**5,
    }


def _higher_kerr_rho21(v: jax.Array, q: jax.Array) -> jax.Array:
    """Pan (29b) spin-dependent rho21 terms at v**4 through v**7."""
    el1 = EULER_GAMMA+math.log(2)+jnp.log(v)
    return ((-865*q**2/1792-405*q**4/2048)*v**4
            +(-98635*q/75264+2031*q**3/7168-1701*q**5/8192)*v**5
            +(-15309*q**6/65536+3897*q**4/16384+9032393*q**2/1806336)*v**6
            +(-72171*q**7/262144+18603*q**5/65536-55169*q**3/16384
              +(107*el1/140-3859374457/1159065600)*q)*v**7)


class SEOBNRv4PublishedFlux:
    """Paper-derived, pre-NQC positive-m moduli and circular energy flux.

    The two required choices document publication ambiguities. ``tabulated``
    ends the ell=8 spin series at v**2, while ``generic-leading-ell8`` also
    applies Pan Eq. (33). The (2,1) choice selects the printed higher Kerr
    spin residual in f or its rho-hybrid diagnostic; neither is claimed to
    reproduce an unexamined implementation. State is (r, phi, pr*, pphi),
    with M=G=c=1 and pphi=|L|/(mu M)>0. Frequencies are M Omega; both
    supplied energies include rest mass. There is no phase, NQC, or merger.
    """

    def __init__(self, lmax: int = 8, *, spin_truncation: str,
                 retain_pan_cubic_spin: bool = False,
                 mode_21_higher_spin: str = "taylor_f",
                 mode_31_higher_spin: str = "taylor_f",
                 mode_32_spin_square: str = "pan_f26",
                 mode_33_higher_spin: str = "taylor_f",
                 mode_43_higher_spin: str = "taylor_f"):
        if type(lmax) is not int or not 2 <= lmax <= 8:
            raise ValueError("lmax must be an integer from 2 through 8")
        if spin_truncation not in {"tabulated", "generic-leading-ell8"}:
            raise ValueError("Choose tabulated or generic-leading-ell8 explicitly")
        if not isinstance(retain_pan_cubic_spin, bool):
            raise TypeError("retain_pan_cubic_spin must be a bool")
        if mode_21_higher_spin not in {"taylor_f", "rho_hybrid"}:
            raise ValueError("mode_21_higher_spin must be taylor_f or rho_hybrid")
        if mode_31_higher_spin not in {"taylor_f", "lal_v4_rho"}:
            raise ValueError("mode_31_higher_spin must be taylor_f or lal_v4_rho")
        if mode_32_spin_square not in {"pan_f26", "lal_v4_zero"}:
            raise ValueError("mode_32_spin_square must be pan_f26 or lal_v4_zero")
        if mode_33_higher_spin not in {"taylor_f", "lal_v4_rho"}:
            raise ValueError("mode_33_higher_spin must be taylor_f or lal_v4_rho")
        if mode_43_higher_spin not in {"taylor_f", "lal_v4_rho"}:
            raise ValueError("mode_43_higher_spin must be taylor_f or lal_v4_rho")
        self.lmax = lmax
        self.spin_truncation = spin_truncation
        self.retain_pan_cubic_spin = retain_pan_cubic_spin
        self.mode_21_higher_spin = mode_21_higher_spin
        self.mode_31_higher_spin = mode_31_higher_spin
        self.mode_32_spin_square = mode_32_spin_square
        self.mode_33_higher_spin = mode_33_higher_spin
        self.mode_43_higher_spin = mode_43_higher_spin
        self.mode_indices = tuple((ell, m) for ell in range(2, lmax+1)
                                  for m in range(1, ell+1))
        self._constants = {mode: abs(_newtonian_constant(*mode))
                           for mode in self.mode_indices}

    @staticmethod
    def _prepare(state, **quantities):
        state = jnp.asarray(state)
        if state.dtype not in (jnp.float32, jnp.float64):
            raise TypeError("state must have float32 or float64 dtype")
        if state.ndim == 0 or state.shape[-1] != 4:
            raise ValueError("state must have trailing dimension (r,phi,pr_star,pphi)")
        keys = tuple(quantities)
        values = [jnp.asarray(quantities[key], dtype=state.dtype) for key in keys]
        *values, j = jnp.broadcast_arrays(*values, state[..., 3])
        args = dict(zip(keys, values))
        valid = jnp.all(jnp.isfinite(state), axis=-1) & (state[..., 0] > 0) & (j > 0)
        for value in args.values():
            valid &= jnp.isfinite(value)
        for key in ("omega", "omega_zero_radial", "heff_over_mu", "hreal_over_M"):
            valid &= args[key] > 0
        a, b = args["x1"], args["x2"]
        valid &= (a >= 0) & (b >= 0) & (jnp.abs(a+b-1) <= 16*jnp.finfo(state.dtype).eps)
        valid &= (jnp.abs(args["chi1"]) <= 1) & (jnp.abs(args["chi2"]) <= 1)
        if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
            raise ValueError("Require finite physical state, frequencies, energies, masses and spins")
        return args, j, valid

    def _signed_amplitudes(self, state, *, omega, omega_zero_radial, heff_over_mu,
                           hreal_over_M, x1, x2, chi1=0., chi2=0.):
        """Real residuals retain smooth second derivatives through mode zeros."""
        args, j, valid = self._prepare(
            state, omega=omega, omega_zero_radial=omega_zero_radial,
            heff_over_mu=heff_over_mu, hreal_over_M=hreal_over_M,
            x1=x1, x2=x2, chi1=chi1, chi2=chi2)
        w, w0, heff, hreal, a, b, s1, s2 = args.values()
        nu, delta, cs, ca = a*b, a-b, (s1+s2)/2, (s1-s2)/2
        v, vphi = jnp.cbrt(w), w*w0**(-2/3)
        rho = _nonspin_rho(v, nu)
        for mode, spin in _spin_rho(v, nu, delta, cs, ca,
                                    spin_truncation=self.spin_truncation).items():
            rho[mode] = rho[mode]+spin
        if self.mode_32_spin_square == "lal_v4_zero":
            r32v1 = -4*nu*cs/(3*(3*nu-1))
            rho[3, 2] = rho[3, 2]+r32v1**2*v**2
        rho[2, 2] = _rho22(v, nu, delta, cs, ca,
                           retain_pan_cubic_spin=self.retain_pan_cubic_spin)
        kerr_spin = (1-2*nu)*cs+delta*ca
        higher_f = _higher_kerr_f(v, kerr_spin)
        if self.mode_21_higher_spin == "rho_hybrid":
            rho[2, 1] = rho[2, 1]+_higher_kerr_rho21(v, kerr_spin)
        if self.mode_33_higher_spin == "lal_v4_rho":
            q = kerr_spin
            rho[3, 3] = (rho[3, 3]+q**2/2*v**4-4*q/3*v**5
                         +5*q**2/36*v**6+(5297*q/2970+q**3/3)*v**7)
        if self.mode_31_higher_spin == "lal_v4_rho":
            q = kerr_spin
            rho[3, 1] = (rho[3, 1]-5*q**2/6*v**4+4*q/9*v**5
                         -49*q**2/108*v**6+(-2579*q/5346+q**3/9)*v**7)
        if self.mode_43_higher_spin == "lal_v4_rho":
            q = kerr_spin
            rho[4, 3] = rho[4, 3]+3*q**2/8*v**4-12113*q/6160*v**5
        result = {}
        for ell, m in self.mode_indices:
            epsilon = (ell+m) % 2
            p = ell+epsilon-1
            c = b**p+(-1)**m*a**p
            residual = c*rho[ell, m]**ell
            if ell <= 4 and m % 2:
                if ((ell, m) != (2, 1) or self.mode_21_higher_spin == "taylor_f") and (
                        (ell, m) != (3, 1) or self.mode_31_higher_spin == "taylor_f") and (
                        (ell, m) != (3, 3) or self.mode_33_higher_spin == "taylor_f") and (
                        (ell, m) != (4, 3) or self.mode_43_higher_spin == "taylor_f"):
                    residual = residual+c*higher_f[ell, m]
                residual += _odd_mass_leading_spin_f(ell, m, v, a, b, s1, s2)
                if (ell, m) == (2, 1):
                    c_over_delta = -sum(a**k*b**(p-1-k) for k in range(p))
                    residual += (c*(427+79*nu)*cs
                                 +c_over_delta*(427+131*nu)*ca)*v**3/84
            source = heff if epsilon == 0 else j*v
            value = (self._constants[ell, m]*nu*vphi**(ell+epsilon)*source
                     *_tail_modulus(ell, m, w, hreal)*residual)
            result[ell, m] = jnp.where(valid, value, jnp.nan)
        return result

    def amplitudes(self, state, **quantities):
        """All positive-m pre-NQC moduli through ``lmax``."""
        return {mode: jnp.abs(value) for mode, value in
                self._signed_amplitudes(state, **quantities).items()}

    def energy_flux(self, state, **quantities):
        """Positive-m sum including the negative-m partners exactly once."""
        signed = self._signed_amplitudes(state, **quantities)
        return emitted_flux_from_modes(quantities["omega"], signed)

    @staticmethod
    def from_hamiltonian(state, hamiltonian, *, omega=None):
        """Supply same-state Hamiltonian energies and orbital frequencies.

        The zero-radial frequency holds radius and angular momentum fixed.
        The local p_phi derivative uses H_eff/mu divided by H_real/M, as in
        the conservative EOM. A summed autodiff objective gives elementwise
        angular partials for parameter batches and trailing trajectory axes.
        """
        state = jnp.asarray(state)
        if state.ndim == 0 or state.shape[-1] != 4:
            raise ValueError("state must have trailing dimension (r,phi,pr_star,pphi)")
        if state.dtype not in (jnp.float32, jnp.float64):
            raise TypeError("state must have float32 or float64 dtype")
        r, pr_star, pphi = state[..., 0], state[..., 2], state[..., 3]
        heff = hamiltonian.effective_hamiltonian(r, pr_star, pphi)
        if heff.shape != pphi.shape:
            raise ValueError("state batch dimensions must include Hamiltonian parameter batches")

        def frequency(radial_momentum):
            effective = hamiltonian.effective_hamiltonian(r, radial_momentum, pphi)
            angular_partial = jax.grad(lambda angular_momentum: jnp.sum(
                hamiltonian.effective_hamiltonian(r, radial_momentum, angular_momentum)
            ))(pphi)
            return angular_partial/hamiltonian.real_hamiltonian_from_eff(effective)

        if omega is None:
            omega = frequency(pr_star)
        raw_omega = jnp.asarray(omega)
        if raw_omega.dtype.kind not in "iuf":
            raise TypeError("omega must have a real numeric dtype")
        try:
            omega = jnp.broadcast_to(jnp.asarray(raw_omega, dtype=state.dtype), heff.shape)
        except ValueError as exc:
            raise ValueError("omega must broadcast to state batch dimensions") from exc
        omega_zero_radial = frequency(jnp.zeros_like(pr_star))

        def parameter(value):
            value = jnp.asarray(value, dtype=state.dtype)
            while value.ndim < heff.ndim:
                value = value[..., None]
            return value

        return dict(omega=omega,
                    omega_zero_radial=jnp.asarray(omega_zero_radial, dtype=state.dtype),
                    heff_over_mu=jnp.asarray(heff, dtype=state.dtype),
                    hreal_over_M=jnp.asarray(hamiltonian.real_hamiltonian_from_eff(heff),
                                             dtype=state.dtype),
                    x1=parameter(hamiltonian.m1), x2=parameter(hamiltonian.m2),
                    chi1=parameter(hamiltonian.chi1), chi2=parameter(hamiltonian.chi2))

    def radiation_reaction_forces(self, omega, pr_star, pphi, *, r, hamiltonian):
        """Return reduced-mass (F_r_star,F_phi), with F_phi=-flux/(nu omega).

        The supplied orbital frequency must be finite and positive.
        """
        dtype = jnp.result_type(r, pr_star, pphi)
        if dtype not in (jnp.float32, jnp.float64):
            raise TypeError("r, pr_star, and pphi must have a real floating dtype")
        r, pr_star, pphi = jnp.broadcast_arrays(
            jnp.asarray(r, dtype=dtype), jnp.asarray(pr_star, dtype=dtype),
            jnp.asarray(pphi, dtype=dtype))
        state = jnp.stack((r, jnp.zeros_like(r), pr_star, pphi), axis=-1)
        args = self.from_hamiltonian(state, hamiltonian, omega=omega)
        nu = args["x1"]*args["x2"]
        valid = (nu > 0) & (pphi > 0) & jnp.isfinite(args["omega"]) & (args["omega"] > 0)
        if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
            raise ValueError("Reduced-mass radiation reaction requires nu>0, p_phi>0, and finite omega>0")
        flux = self.energy_flux(state, **args)
        torque = jnp.where(valid, -flux/(nu*args["omega"]), jnp.nan)
        return pr_star/pphi*torque, torque

    def __call__(self, state, **quantities):
        return self.energy_flux(state, **quantities)


__all__ = ["emitted_flux_from_modes", "SEOBNRv4PublishedFlux"]
