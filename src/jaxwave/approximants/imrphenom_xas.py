"""Clean-room implementation of 4th-generation aligned-spin IMRPhenomXAS waveform model in JAX.

Literature citations:
- Pratten et al. (2020), Phys. Rev. D 102, 064001 (arXiv:2001.11412)
  "Setting the cornerstone for a family of models for gravitational waves from compact binaries: The dominant harmonic for nonprecessing quasicircular black holes"
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
  "IMRPhenomXHM: A multi-mode frequency-domain model for the gravitational wave signal from non-precessing black-hole binaries"
- Khan et al. (2016), Phys. Rev. D 93, 044007 (arXiv:1508.07253)
  "Frequency-domain gravitational waves from nonprecessing black-hole binaries"
- Berti, Cardoso, Will (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
  "Quasinormal modes of Kerr black holes: from astrophysics to string theory"
- Berti, Cardoso, Starinets (2009), Class. Quant. Grav. 26, 163001 (arXiv:0905.2975)
  "Quasinormal modes of black holes and black branes"

Fitting coefficients:
Derived from the official public Mathematica release candidate:
`IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m`
(G. Pratten et al. 2019, Supplementary Material to arXiv:2001.11412; SHA256: 94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52)
Clean-room scoping: Independent implementation from published equations and author fits.
"""

from typing import Optional, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
    component_masses_to_delta,
)

_UNSET = object()


def safe_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    """Autodiff-safe square root that avoids NaN gradients when x -> 0."""
    diff = jnp.maximum(x, 0.0)
    safe_x = jnp.where(diff > 1e-12, diff, 1.0)
    return jnp.where(diff > 1e-12, jnp.sqrt(safe_x), 0.0)


LINB_ETA_COEFFS = [
    3155.163554320194,
    1257.994974060739,
    -32243.28428870636,
    347213.65466879215,
    -1922385.164949654,
    5303591.134694342,
    -5789128.656880263,
]

LINB_S_ETA_COEFFS = [
    [-24.181508118588567, 115.49264174560069, -380.19778216022121],
    [24.725856096415395, -328.37623607519276, 725.60241199889992],
    [23.404604124551920, -646.34101997997209, 1941.8836639528988],
    [-12.814828278938618, -325.92980012408822, 1320.1026401905560],
]

LINB_DELTA_CHI_COEFF = -148.17317525117338

PSI4_ETA_COEFFS = [
    13.393204827580607,
    -175.4248151298946,
    2097.4251161525253,
    -9862.841786379198,
    16026.897939722865,
]

PSI4_S_ETA_COEFFS = [
    [4.7895602776763164, -163.04871764530472, 609.55758504769608],
    [1.3934428041390292, -97.518126812284763, 376.92009325318452],
    [15.649521097877376, 137.33317057388913, -755.95664569064036],
    [13.097315867845779, 149.30405703643268, -764.52421648722589],
]

PSI4_DELTA_CHI_COEFF = 105.37711654943146


def _eval_poly_1d_horner(coeffs: list[float], x: jnp.ndarray) -> jnp.ndarray:
    res = jnp.zeros_like(x)
    for c in reversed(coeffs):
        res = res * x + c
    return res


def compute_phenomx_linb(
    eta: jnp.ndarray,
    s_tot: jnp.ndarray,
    delta_chi: jnp.ndarray,
    delta: jnp.ndarray,
) -> jnp.ndarray:
    p_nospin = _eval_poly_1d_horner(LINB_ETA_COEFFS, eta)
    p_eqspin = jnp.zeros_like(eta)
    s_pow = s_tot
    for row in LINB_S_ETA_COEFFS:
        p_eqspin = p_eqspin + s_pow * _eval_poly_1d_horner(row, eta)
        s_pow = s_pow * s_tot
    return p_nospin + p_eqspin + LINB_DELTA_CHI_COEFF * delta_chi * delta * (eta * eta)


def compute_phenomx_psi4(
    eta: jnp.ndarray,
    s_tot: jnp.ndarray,
    delta_chi: jnp.ndarray,
) -> jnp.ndarray:
    p_nospin = _eval_poly_1d_horner(PSI4_ETA_COEFFS, eta)
    p_eqspin = jnp.zeros_like(eta)
    s_pow = s_tot
    for row in PSI4_S_ETA_COEFFS:
        p_eqspin = p_eqspin + s_pow * _eval_poly_1d_horner(row, eta)
        s_pow = s_pow * s_tot
    diff_eta = jnp.maximum(1.0 - 4.0 * eta, 0.0)
    return (
        p_nospin
        + p_eqspin
        + PSI4_DELTA_CHI_COEFF * delta_chi * safe_sqrt(diff_eta) * (eta * eta)
    )


def pn_phase_canonical_jax(
    mf: jnp.ndarray,
    eta: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    delta: jnp.ndarray,
) -> jnp.ndarray:
    """Canonical 3.5PN TaylorF2 phase expansion per Pratten et al. (2020)."""
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)

    pi_mf = PI * mf
    v = jnp.power(jnp.maximum(pi_mf, 1e-12), 1.0 / 3.0)
    log_v = jnp.log(v)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v
    v7 = v6 * v
    v8 = v7 * v

    c0 = jnp.ones_like(mf)
    c2 = 3715.0 / 756.0 + 55.0 * eta / 9.0
    c3 = -16.0 * PI + 113.0 / 3.0 * delta * chi_a + (113.0 / 3.0 - 76.0 * eta / 3.0) * chi_s

    c4 = (
        15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta**2 / 72.0
        + (-405.0 / 8.0 + 200.0 * eta) * chi_a**2
        - 405.0 / 4.0 * delta * chi_a * chi_s
        + (-405.0 / 8.0 + 5.0 * eta / 2.0) * chi_s**2
    )

    c5 = (
        (38645.0 * PI / 756.0 - 65.0 * PI * eta / 9.0) * (1.0 + 3.0 * log_v)
        + chi_a * (-140.0 * delta * eta / 9.0 - 732985.0 * delta / 2268.0) * (1.0 + 3.0 * log_v)
        + chi_s * (340.0 * eta**2 / 9.0 + 24260.0 * eta / 81.0 - 732985.0 / 2268.0) * (1.0 + 3.0 * log_v)
    )

    gamma_e = 0.57721566490153286
    c6 = (
        11583231236531.0 / 4694215680.0 - 6848.0 * gamma_e / 21.0 - 640.0 * PI**2 / 3.0
        + (-15737765635.0 / 3048192.0 + 2255.0 * PI**2 / 12.0) * eta
        + 76055.0 * eta**2 / 1728.0 - 127825.0 * eta**3 / 1296.0
        + 1135.0 / 3.0 * PI * delta * (2.0 * chi_a)
        + (-480.0 * eta**2 - 263245.0 * eta / 252.0 + 75515.0 / 288.0) * chi_a**2
        + chi_s * (chi_a * (75515.0 * delta / 144.0 - 8225.0 * delta * eta / 18.0) - 520.0 * PI * eta + 2270.0 * PI / 3.0)
        + (1255.0 * eta**2 / 9.0 - 232415.0 * eta / 504.0 + 75515.0 / 288.0) * chi_s**2
        - 13696.0 * math.log(2.0) / 21.0 - 6848.0 / 21.0 * log_v
    )

    c7 = (
        77096675.0 * PI / 254016.0 + 378515.0 * PI * eta / 1512.0 - 74045.0 * PI * eta**2 / 756.0
        + delta * (-25150083775.0 / 3048192.0 + 26804935.0 * eta / 6048.0 - 1985.0 * eta**2 / 48.0) * chi_a
        + (14585.0 * delta / 24.0 - 2380.0 * delta * eta) * chi_a**3
        + (-570.0 * PI + 2240.0 * PI * eta) * chi_a**2
        + chi_s * (
            -25150083775.0 / 3048192.0 + 10566655595.0 * eta / 762048.0 - 1042165.0 * eta**2 / 3024.0 + 5345.0 * eta**3 / 36.0
            - 1140.0 * PI * delta * chi_a + (14585.0 / 8.0 - 7270.0 * eta + 80.0 * eta**2) * chi_a**2
        )
        + chi_s**2 * (-570.0 * PI + 40.0 * PI * eta + delta * (14585.0 / 8.0 - 215.0 * eta / 2.0) * chi_a)
        + chi_s**3 * (14585.0 / 24.0 - 475.0 * eta / 6.0 + 100.0 * eta**2 / 3.0)
    )

    c8 = PI * (
        chi_a * (
            233915.0 * delta / 168.0 - 99185.0 * delta * eta / 252.0
            + (-233915.0 * delta / 168.0 + 99185.0 * delta * eta / 252.0) * 3.0 * log_v
        )
        + chi_s * (
            233915.0 / 168.0 - 3970375.0 * eta / 2268.0 + 19655.0 * eta**2 / 189.0
            + (-233915.0 / 168.0 + 3970375.0 * eta / 2268.0 - 19655.0 * eta**2 / 189.0) * 3.0 * log_v
        )
    )

    pn = c0 + c2 * v2 + c3 * v3 + c4 * v4 + c5 * v5 + c6 * v6 + c7 * v7 + c8 * v8
    return 3.0 / (128.0 * eta) * jnp.power(v, -5.0) * pn - PI / 4.0


pn_phase_canonical = pn_phase_canonical_jax


def d_pn_phase_canonical_jax(
    mf: jnp.ndarray,
    eta: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    delta: jnp.ndarray,
) -> jnp.ndarray:
    """Exact analytical frequency derivative d(phi_tf2)/d(mf) for canonical 3.5PN TaylorF2."""
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)

    pi_mf = PI * mf
    v = jnp.power(jnp.maximum(pi_mf, 1e-12), 1.0 / 3.0)
    log_v = jnp.log(v)
    dv_dmf = (PI / 3.0) * jnp.power(v, -2.0)

    c0 = 1.0
    d_term0 = -5.0 * c0 * jnp.power(v, -6.0)

    c2 = 3715.0 / 756.0 + 55.0 * eta / 9.0
    d_term2 = -3.0 * c2 * jnp.power(v, -4.0)

    c3 = -16.0 * PI + 113.0 / 3.0 * delta * chi_a + (113.0 / 3.0 - 76.0 * eta / 3.0) * chi_s
    d_term3 = -2.0 * c3 * jnp.power(v, -3.0)

    c4 = (
        15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta**2 / 72.0
        + (-405.0 / 8.0 + 200.0 * eta) * chi_a**2
        - 405.0 / 4.0 * delta * chi_a * chi_s
        + (-405.0 / 8.0 + 5.0 * eta / 2.0) * chi_s**2
    )
    d_term4 = -1.0 * c4 * jnp.power(v, -2.0)

    c5_const = (
        (38645.0 * PI / 756.0 - 65.0 * PI * eta / 9.0)
        + chi_a * (-140.0 * delta * eta / 9.0 - 732985.0 * delta / 2268.0)
        + chi_s * (340.0 * eta**2 / 9.0 + 24260.0 * eta / 81.0 - 732985.0 / 2268.0)
    )
    d_term5 = c5_const * 3.0 * jnp.power(v, -1.0)

    gamma_e = 0.57721566490153286
    c6_const = (
        11583231236531.0 / 4694215680.0 - 6848.0 * gamma_e / 21.0 - 640.0 * PI**2 / 3.0
        + (-15737765635.0 / 3048192.0 + 2255.0 * PI**2 / 12.0) * eta
        + 76055.0 * eta**2 / 1728.0 - 127825.0 * eta**3 / 1296.0
        + 1135.0 / 3.0 * PI * delta * (2.0 * chi_a)
        + (-480.0 * eta**2 - 263245.0 * eta / 252.0 + 75515.0 / 288.0) * chi_a**2
        + chi_s * (chi_a * (75515.0 * delta / 144.0 - 8225.0 * delta * eta / 18.0) - 520.0 * PI * eta + 2270.0 * PI / 3.0)
        + (1255.0 * eta**2 / 9.0 - 232415.0 * eta / 504.0 + 75515.0 / 288.0) * chi_s**2
        - 13696.0 * math.log(2.0) / 21.0
    )
    c6_log = -6848.0 / 21.0
    d_term6 = c6_const + c6_log * (log_v + 1.0)

    c7 = (
        77096675.0 * PI / 254016.0 + 378515.0 * PI * eta / 1512.0 - 74045.0 * PI * eta**2 / 756.0
        + delta * (-25150083775.0 / 3048192.0 + 26804935.0 * eta / 6048.0 - 1985.0 * eta**2 / 48.0) * chi_a
        + (14585.0 * delta / 24.0 - 2380.0 * delta * eta) * chi_a**3
        + (-570.0 * PI + 2240.0 * PI * eta) * chi_a**2
        + chi_s * (
            -25150083775.0 / 3048192.0 + 10566655595.0 * eta / 762048.0 - 1042165.0 * eta**2 / 3024.0 + 5345.0 * eta**3 / 36.0
            - 1140.0 * PI * delta * chi_a + (14585.0 / 8.0 - 7270.0 * eta + 80.0 * eta**2) * chi_a**2
        )
        + chi_s**2 * (-570.0 * PI + 40.0 * PI * eta + delta * (14585.0 / 8.0 - 215.0 * eta / 2.0) * chi_a)
        + chi_s**3 * (14585.0 / 24.0 - 475.0 * eta / 6.0 + 100.0 * eta**2 / 3.0)
    )
    d_term7 = 2.0 * c7 * v

    c8_const = PI * (
        chi_a * (233915.0 * delta / 168.0 - 99185.0 * delta * eta / 252.0)
        + chi_s * (233915.0 / 168.0 - 3970375.0 * eta / 2268.0 + 19655.0 * eta**2 / 189.0)
    )
    c8_log = PI * (
        chi_a * (-233915.0 * delta / 168.0 + 99185.0 * delta * eta / 252.0)
        + chi_s * (-233915.0 / 168.0 + 3970375.0 * eta / 2268.0 - 19655.0 * eta**2 / 189.0)
    )
    d_term8 = 3.0 * (v**2) * (c8_const + c8_log * (3.0 * log_v + 1.0))

    d_terms_sum = d_term0 + d_term2 + d_term3 + d_term4 + d_term5 + d_term6 + d_term7 + d_term8
    dphi_dv = (3.0 / (128.0 * eta)) * d_terms_sum
    return dphi_dv * dv_dmf


d_pn_phase_canonical = d_pn_phase_canonical_jax


def compute_spin_variables(
    eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute spin variables S_tot_r and chi_pn_hat per Pratten et al. (2020)."""
    diff = 1.0 - 4.0 * eta
    delta = safe_sqrt(diff)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)
    s_tot_r = (m1**2 * chi1 + m2**2 * chi2) / jnp.maximum(m1**2 + m2**2, 1e-12)
    chi_eff = m1 * chi1 + m2 * chi2
    chi_pn = chi_eff - (76.0 / 113.0) * 0.5 * (chi1 + chi2) * eta
    chi_pn_hat = chi_pn / jnp.maximum(1.0 - (76.0 * eta / 113.0), 1e-12)
    return delta, m1, m2, s_tot_r, chi_pn_hat


class XASParameterFits:
    """28 Hierarchical Parameter Space Fits from Pratten et al. (2020)."""

    @staticmethod
    def fit_f_meco(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (0.018744340279608845 + 0.0077903147004616865 * eta + 0.003940354686136861 * eta**2 - 0.00006693930988501673 * eta**3)/(1.0 - 0.10423384680638834 * eta) + (S**3 * (-0.0028303320109541955 - 0.01319368225506923 * eta - 0.014331096768277578 * eta**2 + 0.00015144033978081733 * eta**3 - 0.865598311695623 * eta**4) + S * (0.0101926920659295 + 0.009158503243006886 * eta + 0.013708626030328923 * eta**2 - 0.020320785712379934 * eta**3 - 0.025528737356553345 * eta**4) + S**2 * (-0.0009694741018756255 - 0.0003933817108395515 * eta - 0.00037579451118948445 * eta**2 + 0.0011504883794752892 * eta**3 + 0.8394906079170564 * eta**4))/(1.0 + S**2 * (-0.22006798741315858 - 1.0787955021207019 * eta - 0.7365550694256703 * eta**2 - 0.07337456927445352 * eta**3 - 95.6841985329734 * eta**4) + S**3 * (0.13153712995215916 + 0.5316213737412727 * eta + 0.47333562697783144 * eta**2 - 0.2879144067301339 * eta**3 + 37.50017276813759 * eta**4) + S * (-0.5471477439918454 - 0.4661075156477155 * eta + 0.94000545001549 * eta**2 + 0.19300625361069154 * eta**3 + 55.72974629895666 * eta**4)) - 0.028476219509487793 * s_delta * (1.0 - 2.8320701810535875 * eta) * eta**2 * (chi1 - chi2) - 0.0011713735642446144 * S * s_delta * eta * (1.0 + 4.905810747693807 * eta) * (chi1 - chi2) + 0.00034375176678815234 * (1.0 + 0.047545158629336504 * eta) * eta**2 * (chi1 - chi2)**2

    @staticmethod
    def fit_final_spin(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        den = jnp.maximum(0.25 * (1.0 - s_delta)**2 + 0.25 * (1.0 + s_delta)**2, 1e-12)
        num_s = 0.25 * (1.0 + s_delta)**2 * chi1 + 0.25 * (1.0 - s_delta)**2 * chi2
        ratio_s = num_s / den
        return (m1**2 * chi1 + m2**2 * chi2) + (2.0 * math.sqrt(3.0) * eta + 20.0830030082033 * eta**2 - 12.333573402277912 * eta**3)/(1.0 + 7.2388440419467335 * eta) + 0.3223660562764661 * s_delta * eta**2 * (1.0 + 9.332575956437443 * eta) * (chi1 - chi2) - 0.059808322561702126 * eta**3 * (chi1 - chi2)**2 + (2.3170397514509933 * s_delta * (1.0 - 3.2624649875884852 * eta) * eta**3 * (chi1 - chi2) * ratio_s) + (((-0.8561951310209386 * eta - 0.09939065676370885 * eta**2 + 1.668810429851045 * eta**3) * ratio_s + (0.5881660363307388 * eta - 2.149269067519131 * eta**2 + 3.4768263932898678 * eta**3) * ratio_s**2 + (0.142443244743048 * eta - 0.9598353840147513 * eta**2 + 1.9595643107593743 * eta**3) * ratio_s**3)/(1.0 + (-0.9142232693081653 + 2.3191363426522633 * eta - 9.710576749140989 * eta**3) * ratio_s))

    @staticmethod
    def fit_final_spin_nnlo(
        eta: jnp.ndarray,
        chi1: jnp.ndarray,
        chi2: jnp.ndarray,
        chi_p: jnp.ndarray,
    ) -> jnp.ndarray:
        """NNLO remnant from the orthogonal-spin prescription."""
        a_parallel = XASParameterFits.fit_final_spin(eta, chi1, chi2)
        mass_discriminant = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        safe_discriminant = jnp.where(
            mass_discriminant > 0.0, mass_discriminant, 1.0
        )
        sqrt_discriminant = jnp.where(
            mass_discriminant > 0.0,
            jnp.sqrt(safe_discriminant),
            0.0,
        )
        m1_norm = 0.5 * (1.0 + sqrt_discriminant)
        s_perp = chi_p * (m1_norm ** 2)
        magnitude = jnp.linalg.norm(
            jnp.stack((a_parallel, s_perp), axis=-1), axis=-1
        )
        signed = jnp.where(a_parallel < 0.0, -magnitude, magnitude)
        return jnp.where(chi_p == 0.0, a_parallel, signed)

    @staticmethod
    def fit_erad(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (((1.0 - (2.0 * math.sqrt(2.0))/3.0) * eta + 0.5609904135313374 * eta**2 - 0.84667563764404 * eta**3 + 3.145145224278187 * eta**4) * (1.0 + S**3 * (-0.6320191645391563 + 4.952698546796005 * eta - 10.023747993978121 * eta**2) + S**2 * (-0.17762802148331427 + 2.176667900182948 * eta**2) + S * (-0.13084389181783257 - 1.1387311580238488 * eta + 5.49074464410971 * eta**2)))/(1.0 + S * (-0.9919475346968611 + 0.367620218664352 * eta + 4.274567337924067 * eta**2)) - 0.01978238971523653 * S * (1.0 - 4.91667749015812 * eta) * s_delta * eta * (chi1 - chi2) - 0.09803730445895877 * s_delta * (1.0 - 3.2283713377939134 * eta) * eta**2 * (chi1 - chi2) + 0.01118530335431078 * eta**3 * (chi1 - chi2)**2

    @staticmethod
    def fit_re_omega(chi: jnp.ndarray) -> jnp.ndarray:
        return (0.05947169566573468 - 0.14989771215394762 * chi + 0.09535606290986028 * chi**2 + 0.02260924869042963 * chi**3 - 0.02501704155363241 * chi**4 - 0.005852438240997211 * chi**5 + 0.0027489038393367993 * chi**6 + 0.0005821983163192694 * chi**7)/(1.0 - 2.8570126619966296 * chi + 2.373335413978394 * chi**2 - 0.6036964688511505 * chi**4 + 0.0873798215084077 * chi**6)

    @staticmethod
    def fit_im_omega(chi: jnp.ndarray) -> jnp.ndarray:
        return (0.014158792290965177 - 0.036989395871554566 * chi + 0.026822526296575368 * chi**2 + 0.0008490933750566702 * chi**3 - 0.004843996907020524 * chi**4 - 0.00014745235759327472 * chi**5 + 0.0001504546201236794 * chi**6)/(1.0 - 2.5900842798681376 * chi + 1.8952576220623967 * chi**2 - 0.31416610693042507 * chi**4 + 0.009002719412204133 * chi**6)

    @staticmethod
    def fit_rd_phase_v4(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (-85.86062966719405 - 4616.740713893726 * eta - 4925.756920247186 * eta**2 + 7732.064464348168 * eta**3 + 12828.269960300782 * eta**4 - 39783.51698102803 * eta**5)/(1.0 + 50.206318806624004 * eta) + 1.0/(1.0 - 0.7296287086910567 * S) * (S**5 * (-1.2063256175329689 - 71.27762576872973 * eta + 525.2433260555774 * eta**2 - 3694.966448470397 * eta**4) + S * (-24.3227986251876 + 50.486023013297824 * eta - 68.32357318404479 * eta**2 - 784.9846285170445 * eta**4) + S**4 * (-1.6801247900898901 - 22.787218950338005 * eta + 76.14492931844407 * eta**2 + 443.82596298646126 * eta**4) + S**2 * (26.62429171331935 - 19.393782361017912 * eta + 13.267927535346725 * eta**2 - 1092.5128972959326 * eta**3 + 2512.1286891850964 * eta**4) + S**3 * (2.7988319050960873 + 11.23611485713071 * eta - 308.98946497905865 * eta**2 + 74.22317145570902 * eta**3 + 3103.82205886183 * eta**4)) + 22.363215261437862 * s_delta * eta * (1.0 + 6.979410949083638 * eta) * (chi1 - chi2)

    @staticmethod
    def fit_rd_phase_d12(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (105.20788785973124 * eta - 180.6010479828692 * eta**2 + 888.4400102515314 * eta**3)/(1.0 + 8.902372184046389 * eta - 22.624125956853074 * eta**2 + 145.96004727185382 * eta**3) + 1.0/(1.0 - 0.6628745847248266 * S) * (S * (0.06519048552628343 - 25.25397971063995 * eta + 58.59408241189781 * eta**2 + 160.14971486043524 * eta**3 - 308.62513664956975 * eta**4) + S**3 * (0.20035146870472367 - 0.28745205203100666 * eta - 47.56042058800358 * eta**4) + S**5 * (-0.2697933899920511 + 4.917070939324979 * eta - 11.61488280763592 * eta**2 - 22.384949087140086 * eta**4) + S**4 * (5.7756520242745735 * eta - 43.97332874253772 * eta**2 + 338.7263666984089 * eta**4) + S**2 * (-5.215945111216946 * eta + 153.95945758807616 * eta**2 - 693.0504179144295 * eta**3 + 835.1725103648205 * eta**4)) - 23.504907495268824 * s_delta * eta**2 * (chi1 - chi2)

    @staticmethod
    def fit_rd_phase_d24(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (141.0810872141363 * eta - 140.61969701245934 * eta**2 - 964.7406436083945 * eta**3)/(1.0 + 1.489446079834234 * eta - 14.913035142611465 * eta**2) + (S**2 * (-62.997389062600035 * eta**2 + 808.457330742532 * eta**3 - 2177.554979351284 * eta**4) + S**3 * (0.04497628581617564 + 64.42947340363101 * eta**2 - 1031.846477275069 * eta**4) + S * (17.36495157980372 * eta - 191.00932194869588 * eta**2 + 930.3458437154668 * eta**3 - 774.3633787391745 * eta**4))/(1.0 - 0.7267610313751913 * S) - 36.66374091965371 * s_delta * (1.0 - 2.4985114985688504 * eta) * eta**2 * (chi1 - chi2)

    @staticmethod
    def fit_rd_phase_d34(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (20.78926405455721 * eta - 34.0584882353451 * eta**2)/(1.0 - 2.4438124009563715 * eta) + (S**4 * (-13.743271480938104 * eta + 215.78111099820157 * eta**2 - 633.1966645925428 * eta**3) + S * (1.7280582989361533 * eta + 73.8367329022058 * eta**2 - 27.976989112929353 * eta**3) + S**2 * (0.046849371468156265 + 6.404060932334562 * eta**3) + S**3 * (18.41570325463385 * eta - 95.57802408341716 * eta**2 + 109.04824706217418 * eta**3))/(1.0 - 0.6862449113932192 * S) + 641.8965762829259 * s_delta * eta**5 * (chi1 - chi2)

    @staticmethod
    def fit_rd_phase_d54(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (27.115659682906863 * eta + 86.27786287161335 * eta**2 + 458.9062853034756 * eta**3)/(1.0 + 3.842206776152707 * eta) + (S * (-7.9407123129681425 * eta + 134.88158268621922 * eta**2 - 316.26970506215554 * eta**3) + S**2 * (9.486783128047414 * eta - 56.05992404859163 * eta**2 + 90.31815139272628 * eta**3))/(1.0 - 0.7162058321905909 * S) + 43.82713604567481 * s_delta * eta**3 * (chi1 - chi2)

    @staticmethod
    def fit_int_phase_v3m_rdv4(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (1126.192427433476 * eta + 49971.74098746552 * eta**2 - 151223.6768602053 * eta**3 + 165076.56749477904 * eta**4)/(1.0 + 171.61217161845025 * eta - 219.68661343717181 * eta**2) + (S**2 * (0.6231339367834515 + 352.35779776330685 * eta**2 - 1216.9113947539156 * eta**3) + S**3 * (1.8572098220199154 + 5.314000623130095 * eta - 146.83065579241452 * eta**3) + S * (-0.1012039136028863 - 48.20617033143495 * eta + 611.6690734410048 * eta**3))/(1.0 - 0.6532315504175634 * S) + 2993.3598520496153 * s_delta * eta**6 * (chi1 - chi2)

    @staticmethod
    def fit_int_phase_v2m_rdv4(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (5922.569864228316 * eta + 602389.5594011653 * eta**2 + 3.7640340576735158e6 * eta**3)/(1.0 + 679.1804843216742 * eta + 10958.704454953331 * eta**2 + 5951.2947288393925 * eta**3) + (S**2 * (5.698079566469573 + 1666.1885442623043 * eta**2 - 4822.805948280586 * eta**3) + S**3 * (0.12180900324633681 - 49.747709066193856 * eta + 60.310245458525685 * eta**3) + S * (6.303589009894369 - 235.25353286241696 * eta + 1190.8227608409995 * eta**3))/(1.0 - 0.9342533635272585 * S + 0.33719529332509046 * S**2) + 356.74395864902294 * s_delta * eta**2 * (chi1 - chi2) + 1693.326644293169 * S * s_delta * eta**4 * (chi1 - chi2)

    @staticmethod
    def fit_int_phase_v2(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (-82.54500000000004 - 5.58197349185435e6 * eta - 3.5225742421184325e8 * eta**2 + 1.4667258334378073e9 * eta**3)/(1.0 + 66757.12830903867 * eta + 5.385164380400193e6 * eta**2 + 2.5176585751772933e6 * eta**3) + 1.0/(1.0 - 0.8701509070254784 * S) * (S**4 * (-5.177484941416572 - 25.296842861196225 * eta + 563.6928051341912 * eta**2 - 1651.1500106393162 * eta**3) + S**2 * (31.383395109529374 + 115.62928454134006 * eta + 93.46377341300065 * eta**2 - 1002.7844776356748 * eta**3) + S**3 * (0.7494310966563801 + 15.377057130883575 * eta - 361.73347121847803 * eta**3) + S**5 * (-4.33748476948353 - 32.59778212915618 * eta + 753.7974313753849 * eta**3) + S * (-16.895476355144673 - 180.72945628412236 * eta - 148.77146757915585 * eta**2 + 1187.8865105889788 * eta**3)) + 515.9898508588834 * S * s_delta * eta**3 * (chi1 - chi2) + 7343.130973149263 * s_delta * (1.0 - 2.7899288785686136 * eta) * eta**3 * (chi1 - chi2)

    @staticmethod
    def fit_int_phase_d43(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (0.4248820426833804 - 906.746595921514 * eta - 282820.39946006844 * eta**2 - 967049.2793750163 * eta**3 + 670077.5414916876 * eta**4)/(1.0 + 1670.9440812294847 * eta + 19783.077247023448 * eta**2) + 1.0/(1.0 - 0.8037103306872213 * S) * (S**3 * (-0.38725361227427885 + 49.164956405737726 * eta - 756.8519869409022 * eta**2 + 3997.8072248553326 * eta**3 - 7028.6084180951 * eta**4) + S**2 * (-0.9135448758751145 - 216.9402989048149 * eta**2 + 705.5621487950621 * eta**3 + 183.05885330208667 * eta**4) + S * (-0.183360658260814 - 10.320162679182692 * eta + 572.1237749921615 * eta**2 - 2812.5299662423513 * eta**3 + 3970.0416726126005 * eta**4)) - 514.8494071830514 * s_delta * (1.0 - 2.900625093731255 * eta) * eta**3 * (chi1 - chi2)

    @staticmethod
    def fit_ins_phase_v3_104(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        poly = (15415.000000000007 + 873401.6255736464 * eta + 376665.64637025696 * eta**2 - 3.9719980569125614e6 * eta**3 + 8.913612508054944e6 * eta**4)/(1.0 + 46.83697749859996 * eta)
        num = (S * (-41128.99121975055 + 103900.17020301468 * eta + 204908.3295566087 * eta**2 - 481840.2512293736 * eta**3)
               + S**2 * (21412.439160773596 - 127668.16906015616 * eta - 21533.26033634209 * eta**2 + 60432.60921041872 * eta**3)
               + S**3 * (13504.791956210198 + 18081.649463745198 * eta - 92538.19022114378 * eta**2 + 174604.23324041915 * eta**3))
        den = 1.0 - 0.3700456424871992 * S - 0.26146258835002273 * S**2 - 0.10335165064705538 * S**3
        asym = (47752.17032707405 * S * s_delta * eta * (chi1 - chi2)
                - 24708.109411857182 * s_delta * (1.0 - 0.9998046496253625 * eta) * eta * (chi1 - chi2)
                - 1296.9289110696955 * eta * (chi1 - chi2)**2)
        return poly + num / den + asym

    @staticmethod
    def fit_ins_phase_d13_104(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        poly = (-17294.000000000007 - 19943.076428555978 * eta + 483033.0998073767 * eta**2)/(1.0 + 4.460294035404433 * eta)
        num = (15977.134017269265 * S**5
               + S**2 * (33147.25683264146 + 30466.10848184382 * eta - 371521.74016586377 * eta**2)
               + S * (33500.56158367473 - 28646.152741568898 * eta - 72683.40959959755 * eta**2)
               + S**3 * (-1067.6298297538203 + 8964.055307425411 * eta - 67599.8180293407 * eta**2)
               + S**4 * (9652.630123774226 - 148504.13863401648 * eta + 527403.5247742739 * eta**2))
        den = 1.0 + 0.48988438820153224 * S
        asym = 12017.062595934838 * s_delta * eta * (chi1 - chi2)
        return poly + num / den + asym

    @staticmethod
    def fit_ins_phase_d23_104(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        poly = (-7579.300000000004 - 120297.86185566607 * eta + 1.1694356931282217e6 * eta**2 - 557253.0066989232 * eta**3)/(1.0 + 18.53018618227582 * eta)
        num = (S**3 * (29103.18384520831 - 83201.347573164 * eta + 262407.5466511699 * eta**2 - 556365.2985387266 * eta**3)
               + S**2 * (43478.71192391914 - 103093.04638355851 * eta + 32933.786598761995 * eta**2 + 213637.97730777826 * eta**3)
               + S * (17783.93150116755 - 33247.42700719334 * eta - 98488.25760862947 * eta**2 + 389715.1720404196 * eta**3))
        den = 1.0 + 2.0108047507657156 * S + 0.7419567859389051 * S**2 - 0.6564911645704183 * S**3
        asym = 3843.083992827935 * s_delta * eta * (chi1 - chi2)
        return poly + num / den + asym

    @staticmethod
    def fit_ins_phase_d43_104(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        poly = (2439.000000000001 - 31133.52170083207 * eta + 28867.73328134167 * eta**2)/(1.0 + 0.41143032589262585 * eta)
        num = (S * (-4310.811205784324 + 22028.99761552467 * eta - 61330.14992532162 * eta**2 + 100525.85671979685 * eta**3)
               + S**2 * (-2637.8441241924743 + 6912.6434374319915 * eta - 31405.568787158234 * eta**2 + 103315.67513958282 * eta**3))
        den = 1.0 - 0.06765889899205788 * S - 0.2674854668199371 * S**2
        asym = 194.5554531509207 * s_delta * eta * (chi1 - chi2)
        return poly + num / den + asym

    @staticmethod
    def fit_rd_amp_v1(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (0.03689164742964719 + 25.417967754401182 * eta + 162.52904393600332 * eta**2)/(1.0 + 61.19874463331437 * eta - 29.628854485544874 * eta**2) + 1.0/(1.0 - 0.7061712786224095 * S) * (S**3 * (-0.141004080435711 - 4.361351839814927 * eta + 33.88984111814397 * eta**2 - 57.855959010022126 * eta**3) + S**4 * (0.037377032197484866 + 0.20299180950548312 * eta + 3.6044803067540183 * eta**2 - 21.53253800848744 * eta**3) + S**2 * (-0.018612493617899906 - 0.7848945943652349 * eta + 1.693629940464679 * eta**2 + 2.649202944790132 * eta**3) + S * (0.10135328197996034 + 3.3931773240261367 * eta - 19.543610795533272 * eta**2 + 34.12180954678677 * eta**3) + S**5 * (0.12815084497386642 + 6.3490731834449345 * eta - 50.90205546408008 * eta**2 + 93.76529966357677 * eta**3)) - 0.04426571511345366 * s_delta * eta**2 * (chi1 - chi2)

    @staticmethod
    def fit_rd_amp_gamma2(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (0.8312293675316895 + 7.480371544268765 * eta - 18.256121237800397 * eta**2)/(1.0 + 10.915453595496611 * eta - 30.578409433912874 * eta**2) + (S * (0.15136992284561207 - 0.037837484199517595 * eta + 0.06523602736595462 * eta**2) + S**2 * (0.008214660865068921 - 0.7347334045130285 * eta + 1.2082315610784062 * eta**2))/(1.0 - 0.8813747076716566 * S + 0.25789638030057566 * S**2) - 0.00548054788508203 * s_delta * eta * (chi1 - chi2)

    @staticmethod
    def fit_rd_amp_gamma3(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        return S**2 * (-0.05633734476062242 + 0.4168560229353532 * eta - 0.6330998412809531 * eta**2) + S * (0.07179105336478316 - 0.8752427297525086 * eta + 2.331724812782498 * eta**2) + (1.3666000000000007 - 4.091333144596439 * eta + 2.109081209912545 * eta**2 - 4.222259944408823 * eta**3)/(1.0 - 2.7440263888207594 * eta)

    @staticmethod
    def fit_int_amp_va(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = s_tot_r
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (1.4873184918202145 + 1974.6112656679577 * eta + 27563.641024162127 * eta**2 - 19837.908020966777 * eta**3)/(1.0 + 143.29004876335128 * eta + 458.4097306093354 * eta**2) + (S**2 * (0.2581300732787427 - 20.585173470211412 * eta + 238.10747802180947 * eta**2 - 550.5211729822532 * eta**3) + S**3 * (-1.5330947512293267 + 117.53109774114579 * eta**3) + S * (2.210151840454939 - 28.903614896319482 * eta + 127.47796975276957 * eta**3))/(1.0 - 0.833383399912328 * S + 0.0790674746971072 * S**2) - 0.016404056649860943 * s_delta * eta**2 * (1.0 + 18073.173360917353 * eta) * (chi1 - chi2)

    @staticmethod
    def fit_int_amp_v2(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (2.2436523786378983 + 2162.4749081764216 * eta + 24460.158604784723 * eta**2 - 12112.140570900956 * eta**3)/(1.0 + 120.78623282522702 * eta + 416.4179522274108 * eta**2) + (S**2 * (-1.6646004496154818 + 39.690635399351976 * eta - 72.25797935657836 * eta**2) + S * (2.0740930179437536 - 57.61505238282958 * eta + 127.67945689822587 * eta**2))/(1.0 - 1.2256937428769066 * S + 0.3083001769574955 * S**2) - 59.52510939953099 * s_delta * (1.0 - 0.22052532962015753 * eta) * eta**2 * (chi1 - chi2)

    @staticmethod
    def fit_int_amp_v3(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (1.195392410912163 + 1677.2558976605421 * eta + 24838.37133975971 * eta**2 - 17277.938868280915 * eta**3)/(1.0 + 144.78606839716073 * eta + 428.8155916011666 * eta**2) + (S**3 * (-0.8003280997506247 + 7.795342953446871 * eta - 13.184393749315833 * eta**2) + S**2 * (-0.386783853162149 - 0.27672690933853966 * eta + 23.606779769106137 * eta**2) + S * (1.4482201622320323 - 31.52459226147479 * eta + 57.36570415789376 * eta**2))/(1.0 - 0.6762974720861794 * S) - 333.7662575986524 * s_delta * (1.0 - 1.5946715606839124 * eta) * eta**3 * (chi1 - chi2)

    @staticmethod
    def fit_ins_amp_r2(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (-0.015178276424448592 - 0.06098548699809163 * eta + 0.4845148547154606 * eta**2)/(1.0 + 0.09799277215675059 * eta) + (S**2 * (0.01761591799745109 - 0.14404522791467844 * eta**2) + S**3 * (0.04834642258922544 * eta - 0.14189350657140673 * eta**2) + S * (0.02300153747158323 + 0.10495263104245876 * eta**2))/(1.0 - 0.7340448493183307 * S) + 0.0018724905795891192 * s_delta * eta**4 * (1.0 + 18642.946301236665 * eta) * (chi1 - chi2)

    @staticmethod
    def fit_ins_amp_r3(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (-0.058572000924124644 - 1.1970535595488723 * eta + 8.4630293045015 * eta**2)/(1.0 + 15.430818840453686 * eta) + (S**2 * (0.014205902285185442 + 0.15983328269721747 * eta - 0.6217809035345132 * eta**2) + S * (0.06565764662955027 + 0.1549903163554525 * eta - 0.5920753389376068 * eta**2))/(1.0 - 0.7506812446570147 * S) + 0.004389995099201855 * s_delta * eta**4 * (1.0 + 24110.628277402946 * eta) * (chi1 - chi2)

    @staticmethod
    def fit_ins_amp_r4(eta: jnp.ndarray, chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        delta, m1, m2, s_tot_r, chi_pn_hat = compute_spin_variables(eta, chi1, chi2)
        S = chi_pn_hat
        diff = jnp.maximum(1.0 - 4.0 * eta, 0.0)
        s_delta = safe_sqrt(diff)
        return (-0.16212854591357853 + 1.617404703616985 * eta - 3.186012733446088 * eta**2 + 5.629598195000046 * eta**3)/(1.0 + 0.04507019231274476 * eta) + (S * (0.17181513336730567 - 0.7052458802501439 * eta + 3.135888331572798 * eta**2 - 7.011766659007688 * eta**3) + S**2 * (-0.053724666247616855 + 0.8911265203926036 * eta - 3.213198718784471 * eta**2 + 3.263330418001537 * eta**3))/(1.0 - 0.9769624571137318 * S + 0.17086112329779823 * S**2) + 0.05575955418803233 * s_delta * eta**4 * (1.0 + 3746.865071813145 * eta) * (chi1 - chi2)


def solve_inspiral_residual_collocation(
    f_endpoint: jnp.ndarray,
    r2: jnp.ndarray,
    r3: jnp.ndarray,
    r4: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Algebraically equivalent dimensionless collocation for inspiral amplitude residuals.

    Solves the 3x3 pseudo-PN system (Eqs. 68, 70 of Pratten et al. 2020) for residuals
    r2, r3, r4 evaluated at nodes [0.5 * f_endpoint, 0.75 * f_endpoint, 1.0 * f_endpoint]:
        r_i = rho1 * (pi * f_i)^(7/3) + rho2 * (pi * f_i)^(8/3) + rho3 * (pi * f_i)^3
    """
    k1, k2, k3 = 0.5, 0.75, 1.0
    p1, p2, p3 = 7.0 / 3.0, 8.0 / 3.0, 3.0

    K11 = k1 ** p1
    K12 = k1 ** p2
    K13 = k1 ** p3

    K21 = k2 ** p1
    K22 = k2 ** p2
    K23 = k2 ** p3

    K31 = k3 ** p1
    K32 = k3 ** p2
    K33 = k3 ** p3

    detK = K11 * (K22 * K33 - K23 * K32) - K12 * (K21 * K33 - K23 * K31) + K13 * (K21 * K32 - K22 * K31)

    det1 = r2 * (K22 * K33 - K23 * K32) - K12 * (r3 * K33 - K23 * r4) + K13 * (r3 * K32 - K22 * r4)
    det2 = K11 * (r3 * K33 - K23 * r4) - r2 * (K21 * K33 - K23 * K31) + K13 * (K21 * r4 - r3 * K31)
    det3 = K11 * (K22 * r4 - r3 * K32) - K12 * (K21 * r4 - r3 * K31) + r2 * (K21 * K32 - K22 * K31)

    rho_tilde1 = det1 / detK
    rho_tilde2 = det2 / detK
    rho_tilde3 = det3 / detK

    x3 = PI * f_endpoint
    rho1 = rho_tilde1 * jnp.power(x3, -p1)
    rho2 = rho_tilde2 * jnp.power(x3, -p2)
    rho3 = rho_tilde3 * jnp.power(x3, -p3)

    return rho1, rho2, rho3


def solve_intermediate_amplitude_quartic(
    f1: jnp.ndarray,
    f4: jnp.ndarray,
    val1: jnp.ndarray,
    d1: jnp.ndarray,
    val_mid: jnp.ndarray,
    val4: jnp.ndarray,
    d4: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve for generic quartic inverse collocation coefficients on dimensionless coordinate u = (f - f1)/(f4 - f1)."""
    delta_f = jnp.maximum(f4 - f1, 1e-8)

    y1 = 1.0 / jnp.maximum(val1, 1e-12)
    y_mid = 1.0 / jnp.maximum(val_mid, 1e-12)
    y4 = 1.0 / jnp.maximum(val4, 1e-12)

    s1 = -(d1 * (y1 ** 2)) * delta_f
    s4 = -(d4 * (y4 ** 2)) * delta_f

    c0 = y1
    c1 = s1
    c2 = -11.0 * y1 + 16.0 * y_mid - 5.0 * y4 - 4.0 * s1 + s4
    c3 = 18.0 * y1 - 32.0 * y_mid + 14.0 * y4 + 5.0 * s1 - 3.0 * s4
    c4 = -8.0 * y1 + 16.0 * y_mid - 8.0 * y4 - 2.0 * s1 + 2.0 * s4

    return c0, c1, c2, c3, c4


@jax.tree_util.register_pytree_node_class
class XASCarrierState:
    """Precomputed intrinsic fits and collocation coefficients for IMRPhenomXAS in JAX."""

    def __init__(
        self,
        mass1=30.0,
        mass2=30.0,
        chi1=_UNSET,
        chi2=_UNSET,
        spin1z=None,
        spin2z=None,
        f_ref=None,
        final_spin_override=None,
        dtype=jnp.float64,
    ):
        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        if final_spin_override is not None:
            final_spin_override = as_array(final_spin_override, dtype=dtype)

        if spin1z is not None:
            s1z = as_array(spin1z, dtype=dtype)
        elif chi1 is not _UNSET and chi1 is not None:
            s1z = as_array(chi1, dtype=dtype)
        else:
            s1z = as_array(0.0, dtype=dtype)

        if spin2z is not None:
            s2z = as_array(spin2z, dtype=dtype)
        elif chi2 is not _UNSET and chi2 is not None:
            s2z = as_array(chi2, dtype=dtype)
        else:
            s2z = as_array(0.0, dtype=dtype)

        is_batched = any(
            hasattr(t, "ndim") and t.ndim > 0 for t in (m1, m2, s1z, s2z)
        ) or (final_spin_override is not None and hasattr(final_spin_override, "ndim") and final_spin_override.ndim > 0)
        if is_batched:
            broadcast_args = (m1, m2, s1z, s2z)
            if final_spin_override is not None:
                broadcast_args += (final_spin_override,)
            broadcasted = jnp.broadcast_arrays(*broadcast_args)
            m1, m2, s1z, s2z = broadcasted[:4]
            if final_spin_override is not None:
                final_spin_override = broadcasted[4]

        swap_mask = m1 < m2
        m1_eff = jnp.where(swap_mask, m2, m1)
        m2_eff = jnp.where(swap_mask, m1, m2)
        s1_eff = jnp.where(swap_mask, s2z, s1z)
        s2_eff = jnp.where(swap_mask, s1z, s2z)

        self.m1_eff = m1_eff
        self.m2_eff = m2_eff
        self.s1_eff = s1_eff
        self.s2_eff = s2_eff
        self.is_batched = is_batched
        self.dtype = dtype
        self.f_ref = f_ref

        total_mass = m1_eff + m2_eff
        eta = component_masses_to_eta(m1_eff, m2_eff)
        delta = component_masses_to_delta(m1_eff, m2_eff)
        m_sec = total_mass * MSUN_SEC

        self.total_mass = total_mass
        self.eta = eta
        self.delta = delta
        self.m_sec = m_sec

        # 1. Evaluate XAS Parameter Fits
        f_meco = XASParameterFits.fit_f_meco(eta, s1_eff, s2_eff)
        aligned_final_spin = XASParameterFits.fit_final_spin(eta, s1_eff, s2_eff)
        final_spin = aligned_final_spin if final_spin_override is None else final_spin_override
        erad = XASParameterFits.fit_erad(eta, s1_eff, s2_eff)
        m_final = jnp.maximum(1.0 - erad, 0.1)

        f_rd = XASParameterFits.fit_re_omega(final_spin) / m_final
        f_damp = XASParameterFits.fit_im_omega(final_spin) / m_final

        chi_fit = jnp.clip(aligned_final_spin, -0.999, 0.999)
        z1 = 1.0 + jnp.power(1.0 - chi_fit**2, 1.0 / 3.0) * (
            jnp.sign(1.0 + chi_fit) * jnp.power(jnp.abs(1.0 + chi_fit), 1.0 / 3.0)
            + jnp.sign(1.0 - chi_fit) * jnp.power(jnp.abs(1.0 - chi_fit), 1.0 / 3.0)
        )
        z2 = safe_sqrt(3.0 * chi_fit**2 + z1**2)
        r_isco = 3.0 + z2 - jnp.sign(chi_fit) * safe_sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
        f_isco = 1.0 / (PI * (r_isco**1.5 + chi_fit))

        lam = XASParameterFits.fit_rd_amp_gamma2(eta, s1_eff, s2_eff)
        sigma = XASParameterFits.fit_rd_amp_gamma3(eta, s1_eff, s2_eff)
        v1_rd = XASParameterFits.fit_rd_amp_v1(eta, s1_eff, s2_eff)

        diff_lam = jnp.maximum(1.0 - lam**2, 0.0)
        f_peak = jnp.abs(f_rd + f_damp * sigma * (safe_sqrt(diff_lam) - 1.0) / jnp.maximum(lam, 1e-8))

        width_rd = sigma * f_damp
        exp_peak = jnp.exp(jnp.clip(lam * (f_peak - f_rd) / width_rd, -50.0, 30.0))
        a_r = v1_rd * ((f_peak - f_rd)**2 + width_rd**2) / width_rd * exp_peak

        r2 = XASParameterFits.fit_ins_amp_r2(eta, s1_eff, s2_eff)
        r3 = XASParameterFits.fit_ins_amp_r3(eta, s1_eff, s2_eff)
        r4 = XASParameterFits.fit_ins_amp_r4(eta, s1_eff, s2_eff)

        f_trans_amp = f_meco + 0.25 * (f_isco - f_meco)
        f_T_phase = 0.6 * (0.5 * f_rd + f_isco)
        delta_R = 0.03 * (f_T_phase - f_meco)
        f_trans_phase = f_meco - delta_R

        rho1, rho2, rho3 = solve_inspiral_residual_collocation(f_trans_amp, r2, r3, r4)

        # Intermediate amplitude collocation
        vA_amp = XASParameterFits.fit_int_amp_va(eta, s1_eff, s2_eff)
        f1_col = f_trans_amp
        f4_col = f_peak
        x1_pi = PI * f1_col
        pn_at_trans, dpn_trans = IMRPhenomXAS.pn_amp_func(
            eta, s1_eff, s2_eff, x1_pi, delta, return_both=True
        )
        val1 = (
            pn_at_trans
            + rho1 * (x1_pi ** (7.0 / 3.0))
            + rho2 * (x1_pi ** (8.0 / 3.0))
            + rho3 * (x1_pi ** 3.0)
        )
        val4 = v1_rd
        dpseudo_trans = PI * (
            (7.0 / 3.0) * rho1 * (x1_pi ** (4.0 / 3.0))
            + (8.0 / 3.0) * rho2 * (x1_pi ** (5.0 / 3.0))
            + 3.0 * rho3 * (x1_pi ** 2.0)
        )
        d1_val = dpn_trans + dpseudo_trans

        f1_pow = jnp.power(jnp.maximum(f1_col, 1e-8), 7.0 / 6.0)
        f4_pow = jnp.power(jnp.maximum(f4_col, 1e-8), 7.0 / 6.0)
        b1_amp = val1 / f1_pow
        b4_amp = val4 / f4_pow
        db1_amp = (d1_val - (7.0 / 6.0) * val1 / f1_col) / f1_pow

        delta_f_peak_rd = f4_col - f_rd
        d4_shape = -2.0 * delta_f_peak_rd / (delta_f_peak_rd**2 + width_rd**2) - (lam / width_rd)
        d4_val = val4 * d4_shape
        db4_amp = (d4_val - (7.0 / 6.0) * val4 / f4_col) / f4_pow
        b_mid_amp = vA_amp

        c0_amp, c1_amp, c2_amp, c3_amp, c4_amp = solve_intermediate_amplitude_quartic(
            f1_col, f4_col, b1_amp, db1_amp, b_mid_amp, b4_amp, db4_amp
        )

        # Ringdown phase collocation
        f_L_rd = f_T_phase
        f_H_rd = f_rd + 1.25 * f_damp
        delta_rd = f_H_rd - f_L_rd
        fn_rd = [
            f_L_rd,
            f_L_rd + 0.5 * (1.0 - 1.0 / math.sqrt(2.0)) * delta_rd,
            f_L_rd + 0.5 * delta_rd,
            f_rd,
            f_H_rd,
        ]
        v4_rd = -XASParameterFits.fit_rd_phase_v4(eta, s1_eff, s2_eff)
        d12_rd = -XASParameterFits.fit_rd_phase_d12(eta, s1_eff, s2_eff)
        d24_rd = -XASParameterFits.fit_rd_phase_d24(eta, s1_eff, s2_eff)
        d34_rd = -XASParameterFits.fit_rd_phase_d34(eta, s1_eff, s2_eff)
        d54_rd = -XASParameterFits.fit_rd_phase_d54(eta, s1_eff, s2_eff)

        vn_rd = [
            v4_rd + d24_rd + d12_rd,
            v4_rd + d24_rd,
            v4_rd + d34_rd,
            v4_rd,
            v4_rd + d54_rd,
        ]
        A_rd_rows = []
        for fn in fn_rd:
            lorentz_entry = 1.0 / jnp.maximum((fn - f_rd)**2 + f_damp**2, 1e-12)
            row = jnp.stack([
                jnp.ones_like(fn),
                jnp.power(jnp.maximum(fn, 1e-8), -1.0 / 3.0),
                jnp.power(jnp.maximum(fn, 1e-8), -2.0),
                jnp.power(jnp.maximum(fn, 1e-8), -4.0),
                lorentz_entry,
            ], axis=-1)
            A_rd_rows.append(row)
        c_coeffs = jnp.linalg.solve(jnp.stack(A_rd_rows, axis=-2), jnp.stack(vn_rd, axis=-1)[..., None]).squeeze(-1)
        c_rd_val, c1_val, c2_val, c4_val, c0_val = [c_coeffs[..., i] for i in range(5)]

        def eval_dphi_rd_val(f_in: jnp.ndarray) -> jnp.ndarray:
            f_safe = jnp.maximum(f_in, 1e-8)
            poly = (
                c_rd_val
                + c1_val * jnp.power(f_safe, -1.0 / 3.0)
                + c2_val / (f_safe**2)
                + c4_val / (f_safe**4)
            )
            lorentz = c0_val / jnp.maximum((f_safe - f_rd)**2 + f_damp**2, 1e-12)
            return (1.0 / eta) * (poly + lorentz)

        def eval_phi_rd_val(f_in: jnp.ndarray) -> jnp.ndarray:
            f_safe = jnp.maximum(f_in, 1e-8)
            poly = (
                c_rd_val * f_safe
                + 1.5 * c1_val * jnp.power(f_safe, 2.0 / 3.0)
                - c2_val / f_safe
                - (1.0 / 3.0) * c4_val / (f_safe**3)
            )
            lorentz = (c0_val / f_damp) * jnp.arctan2(f_safe - f_rd, f_damp)
            return (1.0 / eta) * (poly + lorentz)

        # Inspiral pseudo-PN phase
        f_L_ins = 0.0026 * jnp.ones_like(f_meco)
        f_H_ins = 1.02 * f_meco
        delta_ins = f_H_ins - f_L_ins
        fn_ins = [
            f_L_ins,
            f_L_ins + 0.25 * delta_ins,
            f_L_ins + 0.75 * delta_ins,
            f_H_ins,
        ]
        v3_ins = XASParameterFits.fit_ins_phase_v3_104(eta, s1_eff, s2_eff)
        d13_ins = XASParameterFits.fit_ins_phase_d13_104(eta, s1_eff, s2_eff)
        d23_ins = XASParameterFits.fit_ins_phase_d23_104(eta, s1_eff, s2_eff)
        d43_ins = XASParameterFits.fit_ins_phase_d43_104(eta, s1_eff, s2_eff)
        vn_ins = [
            v3_ins + d13_ins,
            v3_ins + d23_ins,
            v3_ins,
            v3_ins + d43_ins,
        ]
        A_ins_rows = [
            jnp.stack([
                jnp.ones_like(fn),
                jnp.power(jnp.maximum(fn, 1e-8), 1.0 / 3.0),
                jnp.power(jnp.maximum(fn, 1e-8), 2.0 / 3.0),
                fn,
            ], axis=-1)
            for fn in fn_ins
        ]
        sig_raw = jnp.linalg.solve(jnp.stack(A_ins_rows, axis=-2), jnp.stack(vn_ins, axis=-1)[..., None]).squeeze(-1)
        sig_cal = (-5.0 / (128.0 * (PI ** (5.0 / 3.0)))) * sig_raw
        sig1_cal, sig2, sig3, sig4 = [sig_cal[..., i] for i in range(4)]

        # Intermediate phase collocation
        f_L_int = f_trans_phase
        f_H_int = f_T_phase + 0.5 * delta_R
        delta_int = f_H_int - f_L_int
        fn_int = [
            f_L_int,
            f_L_int + 0.5 * (1.0 - 1.0 / math.sqrt(2.0)) * delta_int,
            f_L_int + 0.5 * delta_int,
            f_L_int + 0.5 * (1.0 + 1.0 / math.sqrt(2.0)) * delta_int,
            f_H_int,
        ]
        v5_int = eta * eval_dphi_rd_val(f_T_phase)
        v3m_rdv4 = -XASParameterFits.fit_int_phase_v3m_rdv4(eta, s1_eff, s2_eff)
        v3_int = v4_rd + v3m_rdv4
        v2m_rdv4 = -XASParameterFits.fit_int_phase_v2m_rdv4(eta, s1_eff, s2_eff)
        v2_direct = -XASParameterFits.fit_int_phase_v2(eta, s1_eff, s2_eff)
        v2_int = 0.75 * (v4_rd + v2m_rdv4) + 0.25 * v2_direct
        d43_int = -XASParameterFits.fit_int_phase_d43(eta, s1_eff, s2_eff)
        v4_int = v3_int + d43_int

        dphi_tf2_trans = d_pn_phase_canonical_jax(f_trans_phase, eta, s1_eff, s2_eff, delta)
        f_trans_safe = jnp.maximum(f_trans_phase, 1e-8)
        d_pseudo_trans_raw = (
            sig1_cal
            + sig2 * jnp.power(f_trans_safe, 1.0 / 3.0)
            + sig3 * jnp.power(f_trans_safe, 2.0 / 3.0)
            + sig4 * f_trans_safe
        )
        v1_int = eta * dphi_tf2_trans + d_pseudo_trans_raw
        vn_int = [v1_int, v2_int, v3_int, v4_int, v5_int]

        A_int_rows = []
        b_int_entries = []
        for i, fn in enumerate(fn_int):
            lorentz_int = 4.0 * c0_val / jnp.maximum((fn - f_rd)**2 + (2.0 * f_damp)**2, 1e-12)
            b_entry = vn_int[i] - lorentz_int
            row = jnp.stack([
                jnp.ones_like(fn),
                jnp.power(jnp.maximum(fn, 1e-8), -1.0),
                jnp.power(jnp.maximum(fn, 1e-8), -2.0),
                jnp.power(jnp.maximum(fn, 1e-8), -3.0),
                jnp.power(jnp.maximum(fn, 1e-8), -4.0),
            ], axis=-1)
            A_int_rows.append(row)
            b_int_entries.append(b_entry)

        b_coeffs = jnp.linalg.solve(jnp.stack(A_int_rows, axis=-2), jnp.stack(b_int_entries, axis=-1)[..., None]).squeeze(-1)
        b0_val, b1_val, b2_val, b3_val, b4_val = [b_coeffs[..., i] for i in range(5)]

        def eval_dphi_int_val(f_in: jnp.ndarray) -> jnp.ndarray:
            f_safe = jnp.maximum(f_in, 1e-8)
            poly = (
                b0_val
                + b1_val / f_safe
                + b2_val / (f_safe**2)
                + b3_val / (f_safe**3)
                + b4_val / (f_safe**4)
            )
            lorentz = 4.0 * c0_val / jnp.maximum((f_safe - f_rd)**2 + (2.0 * f_damp)**2, 1e-12)
            return (1.0 / eta) * (poly + lorentz)

        def eval_phi_int_val(f_in: jnp.ndarray) -> jnp.ndarray:
            f_safe = jnp.maximum(f_in, 1e-8)
            poly = (
                b0_val * f_safe
                + b1_val * jnp.log(f_safe)
                - b2_val / f_safe
                - 0.5 * b3_val / (f_safe**2)
                - (1.0 / 3.0) * b4_val / (f_safe**3)
            )
            lorentz = (2.0 * c0_val / f_damp) * jnp.arctan2(f_safe - f_rd, 2.0 * f_damp)
            return (1.0 / eta) * (poly + lorentz)

        f_join_rd = f_H_int
        dt_rd = eval_dphi_int_val(f_join_rd) - eval_dphi_rd_val(f_join_rd)
        dphi_rd = eval_phi_int_val(f_join_rd) - (eval_phi_rd_val(f_join_rd) + dt_rd * f_join_rd)

        f_fit = f_rd - f_damp
        dpsi_fit = eval_dphi_rd_val(f_fit) + dt_rd
        m1_sq = m1_eff * m1_eff
        m2_sq = m2_eff * m2_eff
        m_sq_sum = jnp.maximum(m1_sq + m2_sq, 1e-12)
        s_tot = (m1_sq * s1_eff + m2_sq * s2_eff) / m_sq_sum
        delta_chi = s1_eff - s2_eff
        linb_val = compute_phenomx_linb(eta, s_tot, delta_chi, delta)
        psi4_val = compute_phenomx_psi4(eta, s_tot, delta_chi)
        t_hybrid = -2.0 * PI * (500.0 + psi4_val)
        tshift = -(linb_val + t_hybrid) - dpsi_fit
        tshift_sec = tshift * m_sec

        d_pseudo_tail = (
            sig2 * jnp.power(f_trans_safe, 1.0 / 3.0)
            + sig3 * jnp.power(f_trans_safe, 2.0 / 3.0)
            + sig4 * f_trans_safe
        )
        sigma_1 = eta * (eval_dphi_int_val(f_trans_phase) - dphi_tf2_trans) - d_pseudo_tail

        def eval_phi_ins_raw(f_in: jnp.ndarray, eta_p, s1_p, s2_p, d_p, sig1_p, sig2_p, sig3_p, sig4_p):
            tf2 = pn_phase_canonical_jax(f_in, eta_p, s1_p, s2_p, d_p)
            f_safe = jnp.maximum(f_in, 1e-8)
            pseudo = (1.0 / eta_p) * (
                sig1_p * f_safe
                + 0.75 * sig2_p * jnp.power(f_safe, 4.0 / 3.0)
                + 0.60 * sig3_p * jnp.power(f_safe, 5.0 / 3.0)
                + 0.50 * sig4_p * jnp.power(f_safe, 2.0)
            )
            return tf2 + pseudo

        phi_ins_trans = eval_phi_ins_raw(f_trans_phase, eta, s1_eff, s2_eff, delta, sigma_1, sig2, sig3, sig4)
        sigma_0 = eta * (eval_phi_int_val(f_trans_phase) - phi_ins_trans)

        # Store precomputed fields
        self.f_trans_amp = f_trans_amp
        self.f_peak = f_peak
        self.f_trans_phase = f_trans_phase
        self.f_join_rd = f_join_rd
        self.f_rd = f_rd
        self.f_damp = f_damp
        self.width_rd = width_rd
        self.lam = lam
        self.a_r = a_r

        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3

        self.c0_amp = c0_amp
        self.c1_amp = c1_amp
        self.c2_amp = c2_amp
        self.c3_amp = c3_amp
        self.c4_amp = c4_amp
        self.f1_col = f1_col
        self.f4_col = f4_col

        self.c_rd_val = c_rd_val
        self.c1_val = c1_val
        self.c2_val = c2_val
        self.c4_val = c4_val
        self.c0_val = c0_val
        self.dt_rd = dt_rd
        self.dphi_rd = dphi_rd

        self.sig0 = sigma_0
        self.sig1 = sigma_1
        self.sig2 = sig2
        self.sig3 = sig3
        self.sig4 = sig4

        self.b0_val = b0_val
        self.b1_val = b1_val
        self.b2_val = b2_val
        self.b3_val = b3_val
        self.b4_val = b4_val

        self.tshift_sec = tshift_sec

    def tree_flatten(self):
        children = (
            self.m1_eff, self.m2_eff, self.s1_eff, self.s2_eff,
            self.total_mass, self.eta, self.delta, self.m_sec,
            self.f_trans_amp, self.f_peak, self.f_trans_phase, self.f_join_rd,
            self.f_rd, self.f_damp, self.width_rd, self.lam, self.a_r,
            self.rho1, self.rho2, self.rho3,
            self.c0_amp, self.c1_amp, self.c2_amp, self.c3_amp, self.c4_amp,
            self.f1_col, self.f4_col,
            self.c_rd_val, self.c1_val, self.c2_val, self.c4_val, self.c0_val,
            self.dt_rd, self.dphi_rd,
            self.sig0, self.sig1, self.sig2, self.sig3, self.sig4,
            self.b0_val, self.b1_val, self.b2_val, self.b3_val, self.b4_val,
            self.tshift_sec,
        )
        aux_data = (self.is_batched, self.dtype, self.f_ref)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (
            obj.m1_eff, obj.m2_eff, obj.s1_eff, obj.s2_eff,
            obj.total_mass, obj.eta, obj.delta, obj.m_sec,
            obj.f_trans_amp, obj.f_peak, obj.f_trans_phase, obj.f_join_rd,
            obj.f_rd, obj.f_damp, obj.width_rd, obj.lam, obj.a_r,
            obj.rho1, obj.rho2, obj.rho3,
            obj.c0_amp, obj.c1_amp, obj.c2_amp, obj.c3_amp, obj.c4_amp,
            obj.f1_col, obj.f4_col,
            obj.c_rd_val, obj.c1_val, obj.c2_val, obj.c4_val, obj.c0_val,
            obj.dt_rd, obj.dphi_rd,
            obj.sig0, obj.sig1, obj.sig2, obj.sig3, obj.sig4,
            obj.b0_val, obj.b1_val, obj.b2_val, obj.b3_val, obj.b4_val,
            obj.tshift_sec,
        ) = children
        obj.is_batched, obj.dtype, obj.f_ref = aux_data
        return obj

    def eval_waveform(
        self,
        f: jnp.ndarray,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phi_c: Union[float, jnp.ndarray] = 0.0,
        t_c: Union[float, jnp.ndarray] = 0.0,
        return_phase: bool = False,
    ):
        f_eval = as_array(f, dtype=self.dtype)
        dist = as_array(distance, dtype=self.dtype)
        inc = as_array(inclination, dtype=self.dtype)
        phi0 = as_array(phi_c, dtype=self.dtype)
        t0 = as_array(t_c, dtype=self.dtype)

        is_batched = self.is_batched or any(
            hasattr(t, "ndim") and t.ndim > 0 for t in (dist, inc, phi0, t0)
        )
        if is_batched:
            f_eval_b = jnp.expand_dims(f_eval, 0) if f_eval.ndim == 1 else f_eval
            m_sec_eval = jnp.expand_dims(self.m_sec, -1) if self.m_sec.ndim == 1 else self.m_sec
            def u(t: jnp.ndarray) -> jnp.ndarray:
                return jnp.expand_dims(t, -1) if (hasattr(t, "ndim") and t.ndim == 1) else t
        else:
            f_eval_b = f_eval
            m_sec_eval = self.m_sec
            def u(t: jnp.ndarray) -> jnp.ndarray:
                return t

        f_eval_safe = jnp.maximum(f_eval_b, 1e-8)
        mf = m_sec_eval * f_eval_safe

        eta_u = u(self.eta)
        s1_u = u(self.s1_eff)
        s2_u = u(self.s2_eff)
        d_u = safe_sqrt(1.0 - 4.0 * eta_u)
        pi_mf = PI * mf

        # 1. Amplitude
        pn_amp = IMRPhenomXAS.pn_amp_func(eta_u, s1_u, s2_u, pi_mf, d_u)
        scaled_amp_ins = (
            pn_amp
            + u(self.rho1) * (pi_mf ** (7.0 / 3.0))
            + u(self.rho2) * (pi_mf ** (8.0 / 3.0))
            + u(self.rho3) * (pi_mf ** 3.0)
        )

        f_rd_u = u(self.f_rd)
        width_rd_u = u(self.width_rd)
        lam_u = u(self.lam)
        a_r_u = u(self.a_r)

        exp_arg_mr = -(lam_u * (mf - f_rd_u)) / width_rd_u
        scaled_amp_mr = (a_r_u * width_rd_u / ((mf - f_rd_u)**2 + width_rd_u**2)) * jnp.exp(jnp.clip(exp_arg_mr, -50.0, 50.0))

        f1_u = u(self.f1_col)
        f4_u = u(self.f4_col)
        delta_f_u = jnp.maximum(f4_u - f1_u, 1e-8)
        u_int = (mf - f1_u) / delta_f_u
        poly_inv = u(self.c0_amp) + u_int * (u(self.c1_amp) + u_int * (u(self.c2_amp) + u_int * (u(self.c3_amp) + u_int * u(self.c4_amp))))
        mf_pow = jnp.power(jnp.maximum(mf, 1e-8), 7.0 / 6.0)
        scaled_amp_int = mf_pow / jnp.maximum(poly_inv, 1e-12)

        scaled_amp = jnp.where(
            mf <= u(self.f_trans_amp),
            scaled_amp_ins,
            jnp.where(
                mf <= u(self.f_peak),
                scaled_amp_int,
                scaled_amp_mr,
            ),
        )

        dist_sec = (u(dist) * MPC_SI) / C_SI
        leading_amp_factor = (
            jnp.sqrt(5.0 * eta_u / 24.0)
            * (m_sec_eval ** (5.0 / 6.0))
            / (PI ** (2.0 / 3.0) * dist_sec)
        )
        full_amp = leading_amp_factor * (f_eval_safe ** (-7.0 / 6.0)) * scaled_amp

        # 2. Phase
        sig1_u = u(self.sig1)
        sig2_u = u(self.sig2)
        sig3_u = u(self.sig3)
        sig4_u = u(self.sig4)
        sig0_u = u(self.sig0)

        b0_u = u(self.b0_val)
        b1_u = u(self.b1_val)
        b2_u = u(self.b2_val)
        b3_u = u(self.b3_val)
        b4_u = u(self.b4_val)

        c_rd_u = u(self.c_rd_val)
        c1_u = u(self.c1_val)
        c2_u = u(self.c2_val)
        c4_u = u(self.c4_val)
        c0_u = u(self.c0_val)
        dt_rd_u = u(self.dt_rd)
        dphi_rd_u = u(self.dphi_rd)
        f_damp_u = u(self.f_damp)

        def eval_grid_phi_ins(f_in: jnp.ndarray) -> jnp.ndarray:
            tf2 = pn_phase_canonical_jax(f_in, eta_u, s1_u, s2_u, d_u)
            f_safe = jnp.maximum(f_in, 1e-8)
            pseudo = (1.0 / eta_u) * (
                sig1_u * f_safe
                + 0.75 * sig2_u * jnp.power(f_safe, 4.0 / 3.0)
                + 0.60 * sig3_u * jnp.power(f_safe, 5.0 / 3.0)
                + 0.50 * sig4_u * jnp.power(f_safe, 2.0)
                + sig0_u
            )
            return tf2 + pseudo

        def eval_grid_phi_int(f_in: jnp.ndarray) -> jnp.ndarray:
            f_safe = jnp.maximum(f_in, 1e-8)
            poly = (
                b0_u * f_safe
                + b1_u * jnp.log(f_safe)
                - b2_u / f_safe
                - 0.5 * b3_u / (f_safe**2)
                - (1.0 / 3.0) * b4_u / (f_safe**3)
            )
            lorentz = (2.0 * c0_u / f_damp_u) * jnp.arctan2(f_safe - u(self.f_rd), 2.0 * f_damp_u)
            return (1.0 / eta_u) * (poly + lorentz)

        def eval_grid_phi_rd(f_in: jnp.ndarray) -> jnp.ndarray:
            f_safe = jnp.maximum(f_in, 1e-8)
            poly = (
                c_rd_u * f_safe
                + 1.5 * c1_u * jnp.power(f_safe, 2.0 / 3.0)
                - c2_u / f_safe
                - (1.0 / 3.0) * c4_u / (f_safe**3)
            )
            lorentz = (c0_u / f_damp_u) * jnp.arctan2(f_safe - u(self.f_rd), f_damp_u)
            return (1.0 / eta_u) * (poly + lorentz) + dt_rd_u * f_safe + dphi_rd_u

        phi_ins_val = eval_grid_phi_ins(mf)
        phi_int_val = eval_grid_phi_int(mf)
        phi_rd_val = eval_grid_phi_rd(mf)

        psi_raw = jnp.where(
            mf <= u(self.f_trans_phase),
            phi_ins_val,
            jnp.where(
                mf <= u(self.f_join_rd),
                phi_int_val,
                phi_rd_val,
            ),
        )

        tshift_u = u(self.tshift_sec)
        if self.f_ref is not None:
            fref_t = as_array(self.f_ref, dtype=self.dtype)
            fref_u = jnp.expand_dims(fref_t, -1) if (is_batched and f_eval_b.ndim > fref_t.ndim) else fref_t
            mf_ref = m_sec_eval * fref_u
            psi_ref = jnp.where(
                mf_ref <= u(self.f_trans_phase),
                eval_grid_phi_ins(mf_ref),
                jnp.where(
                    mf_ref <= u(self.f_join_rd),
                    eval_grid_phi_int(mf_ref),
                    eval_grid_phi_rd(mf_ref),
                ),
            )
            total_phase = (
                psi_raw
                - psi_ref
                + tshift_u * (f_eval_b - fref_u)
                + 2.0 * PI * f_eval_b * u(t0)
                - 2.0 * u(phi0)
                + 3.0 * PI / 4.0
            )
        else:
            total_phase = (
                psi_raw
                + tshift_u * f_eval_b
                + 2.0 * PI * f_eval_b * u(t0)
                - 2.0 * u(phi0)
                + 3.0 * PI / 4.0
            )

        cos_inc = jnp.cos(u(inc))
        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        h_plus = (plus_factor * full_amp) * (jnp.cos(total_phase) - 1j * jnp.sin(total_phase))
        h_cross = (cross_factor * full_amp) * (-jnp.sin(total_phase) - 1j * jnp.cos(total_phase))

        h_plus = jnp.where(f_eval_b > 0.0, h_plus, jnp.zeros_like(h_plus))
        h_cross = jnp.where(f_eval_b > 0.0, h_cross, jnp.zeros_like(h_cross))

        if return_phase:
            return h_plus, h_cross, total_phase
        return h_plus, h_cross

    def eval_phase(self, f: jnp.ndarray) -> jnp.ndarray:
        """Evaluate carrier phase without amplitude overhead."""
        return self.eval_waveform(f, inclination=0.0, phi_c=0.0, t_c=0.0, return_phase=True)[2]

    def __call__(self, *args, **kwargs):
        return self.eval_waveform(*args, **kwargs)


@jax.tree_util.register_pytree_node_class
class IMRPhenomXAS:
    """Clean-Room 4th-generation aligned-spin frequency-domain IMRPhenomXAS waveform model in JAX."""

    definition_id = "IMRPhenomXAS"

    def __new__(cls, *args, **kwargs):
        if args or kwargs:
            return cls.evaluate(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self) -> None:
        pass

    def tree_flatten(self):
        return ((), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

    @staticmethod
    def build_state(
        mass1: Union[float, jnp.ndarray] = 30.0,
        mass2: Union[float, jnp.ndarray] = 30.0,
        chi1: Union[float, jnp.ndarray, object] = _UNSET,
        chi2: Union[float, jnp.ndarray, object] = _UNSET,
        spin1z: Optional[Union[float, jnp.ndarray]] = None,
        spin2z: Optional[Union[float, jnp.ndarray]] = None,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        _final_spin_override: Optional[Union[float, jnp.ndarray]] = None,
        dtype=jnp.float64,
        **kwargs,
    ) -> XASCarrierState:
        """Construct precomputed carrier state for reusable evaluations."""
        return XASCarrierState(
            mass1=mass1,
            mass2=mass2,
            chi1=chi1,
            chi2=chi2,
            spin1z=spin1z,
            spin2z=spin2z,
            f_ref=f_ref,
            final_spin_override=_final_spin_override,
            dtype=dtype,
        )

    @staticmethod
    def pn_amp_func(
        eta_u: jnp.ndarray,
        s1_u: jnp.ndarray,
        s2_u: jnp.ndarray,
        pi_mf: jnp.ndarray,
        d_u: jnp.ndarray,
        *,
        derivative: bool = False,
        return_both: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """3PN TaylorF2 amplitude expansion per Pratten et al. (2020) and Mathematica PNAmplitude."""
        t1 = jnp.ones_like(pi_mf)
        t2 = (1.0 / 672.0) * jnp.power(jnp.maximum(pi_mf, 1e-12), 2.0 / 3.0) * (-969.0 + 1804.0 * eta_u)
        t3 = (1.0 / 48.0) * pi_mf * ((81.0 * (1.0 + d_u) - 44.0 * eta_u) * s1_u + (81.0 - 81.0 * d_u - 44.0 * eta_u) * s2_u)
        t4 = (1.0 / 8128512.0) * jnp.power(jnp.maximum(pi_mf, 1e-12), 4.0 / 3.0) * (
            -27312085.0 + 35371056.0 * eta_u**2 - 10287648.0 * (1.0 + d_u) * s1_u**2
            - 10287648.0 * s2_u**2 + 10287648.0 * d_u * s2_u**2
            + 24.0 * eta_u * (-1975055.0 + 857304.0 * s1_u**2 - 994896.0 * s1_u * s2_u + 857304.0 * s2_u**2)
        )
        t5 = (1.0 / 32256.0) * jnp.power(jnp.maximum(pi_mf, 1e-12), 5.0 / 3.0) * (
            42840.0 * PI * (-1.0 + 4.0 * eta_u)
            + (287213.0 * (1.0 + d_u) - 4.0 * (93414.0 + 2083.0 * d_u) * eta_u - 35632.0 * eta_u**2) * s1_u
            - 6048.0 * (-1.0 - d_u + (3.0 + d_u) * eta_u) * s1_u**3
            + s2_u * (-35632.0 * eta_u**2 - (-1.0 + d_u) * (287213.0 + 6048.0 * s2_u**2) + 4.0 * eta_u * (-93414.0 + 2083.0 * d_u + 1512.0 * (-3.0 + d_u) * s2_u**2))
        )
        t6 = -(1.0 / 60085960704.0) * (pi_mf ** 2.0) * (
            -324322727232.0 * eta_u**3
            - 336.0 * eta_u**2 * (-3248849057.0 + 1809550512.0 * s1_u**2 - 2954929824.0 * s1_u * s2_u + 1809550512.0 * s2_u**2)
            + 12.0 * eta_u * (-545384828789.0 + 21384760320.0 * PI**2 + 77616.0 * (-158633.0 + 282718.0 * d_u) * s1_u**2 + 49568837472.0 * s1_u * s2_u - 12312458928.0 * s2_u**2 - 21943440288.0 * d_u * s2_u**2 - 8345272320.0 * PI * (s1_u + s2_u))
            + 7.0 * (177520268561.0 + 29362199328.0 * (1.0 + d_u) * s1_u**2 + 29362199328.0 * s2_u**2 - 29362199328.0 * d_u * s2_u**2 + 12160253952.0 * PI * (s1_u + d_u * s1_u + s2_u - d_u * s2_u))
        )
        amp = t1 + t2 + t3 + t4 + t5 + t6
        if derivative or return_both:
            x_safe = jnp.maximum(pi_mf, 1e-12)
            dpn_dx = ((2.0 / 3.0) * t2 + t3 + (4.0 / 3.0) * t4 + (5.0 / 3.0) * t5 + 2.0 * t6) / x_safe
            dpn_d_mf = PI * dpn_dx
            if return_both:
                return amp, dpn_d_mf
            return dpn_d_mf
        return amp

    @staticmethod
    def evaluate(
        f: Optional[jnp.ndarray] = None,
        mass1: Union[float, jnp.ndarray] = 30.0,
        mass2: Union[float, jnp.ndarray] = 30.0,
        chi1: Union[float, jnp.ndarray, object] = _UNSET,
        chi2: Union[float, jnp.ndarray, object] = _UNSET,
        spin1z: Optional[Union[float, jnp.ndarray]] = None,
        spin2z: Optional[Union[float, jnp.ndarray]] = None,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phi_c: Union[float, jnp.ndarray] = 0.0,
        t_c: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        window_type: str = "planck",
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        sample_frequencies: Optional[jnp.ndarray] = None,
        return_phase: bool = False,
        **kwargs,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Generate frequency-domain polarizations h_plus and h_cross in JAX."""
        freqs = f if f is not None else sample_frequencies
        if freqs is None:
            raise ValueError("Must provide frequency grid via 'f' or 'sample_frequencies'")

        dtype = jnp.float64

        final_spin_override = kwargs.pop("_final_spin_override", None)
        if final_spin_override is None:
            final_spin_override = kwargs.pop("final_spin_override", None)

        s1z_arg = kwargs.pop("spin1z", None) if spin1z is None else spin1z
        s2z_arg = kwargs.pop("spin2z", None) if spin2z is None else spin2z

        if s1z_arg is not None:
            s1z = as_array(s1z_arg, dtype=dtype)
        elif chi1 is not _UNSET and chi1 is not None:
            s1z = as_array(chi1, dtype=dtype)
        else:
            s1z = as_array(0.0, dtype=dtype)

        if s2z_arg is not None:
            s2z = as_array(s2z_arg, dtype=dtype)
        elif chi2 is not _UNSET and chi2 is not None:
            s2z = as_array(chi2, dtype=dtype)
        else:
            s2z = as_array(0.0, dtype=dtype)

        phi0 = kwargs.pop("phic", phi_c)
        if phi0 is None:
            phi0 = 0.0
        t0 = kwargs.pop("tc", t_c)
        if t0 is None:
            t0 = 0.0

        # Optional ignored/compatibility kwargs
        kwargs.pop("f_lower", None)
        kwargs.pop("f_final", None)
        kwargs.pop("delta_f", None)
        kwargs.pop("dtype", None)
        kwargs.pop("window_type", None)
        kwargs.pop("phasing_convention", None)
        kwargs.pop("reference_frame", None)

        carrier = XASCarrierState(
            mass1=mass1,
            mass2=mass2,
            chi1=s1z,
            chi2=s2z,
            f_ref=f_ref,
            final_spin_override=final_spin_override,
            dtype=dtype,
        )
        return carrier.eval_waveform(
            f=freqs,
            distance=distance,
            inclination=inclination,
            phi_c=phi0,
            t_c=t0,
            return_phase=return_phase,
        )

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


__all__ = [
    "IMRPhenomXAS",
    "XASCarrierState",
    "XASParameterFits",
    "compute_phenomx_linb",
    "compute_phenomx_psi4",
    "pn_phase_canonical_jax",
    "d_pn_phase_canonical_jax",
    "pn_phase_canonical",
    "d_pn_phase_canonical",
    "compute_spin_variables",
    "solve_inspiral_residual_collocation",
    "solve_intermediate_amplitude_quartic",
    "safe_sqrt",
]
