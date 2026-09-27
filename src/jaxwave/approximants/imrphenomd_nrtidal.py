"""IMRPhenomD_NRTidal & IMRPhenomD_NRTidalv2 Frequency-Domain Waveform Approximants in JAX.

Tidal formulas from peer-reviewed literature:
- Dietrich, Bernuzzi, Tichy (2017), Phys. Rev. D 96, 121501(R) (arXiv:1706.02969) [NRTidal v1]
- Dietrich et al. (2019), Phys. Rev. D 100, 044003 (arXiv:1905.06011) [NRTidalv2: Eqs. 17-21, 26-29; Eq. 24 independent Pv2 component]
- Dietrich et al. (2018), Phys. Rev. D 99, 024029 (arXiv:1804.02235) [BNS merger frequency fit Sec II.2.2 Eq. 11]
- Yagi & Yunes (2013), Science 341, 365; PRD 88, 023009 [Universal I-Love-Q relations]
- McKechan, Robinson, Sathyaprakash (2010), Class. Quantum Grav. 27, 084020 [Planck taper]
"""

from typing import Optional, Tuple, Union
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    MPC_SI,
    MSUN_SEC,
    PI,
)
from diffgw_core.nrtidal_fits import (
    NRTIDAL_V1_PADÉ_COEFFS,
    NRTIDAL_V2_PHASE_COEFFS,
    NRTIDAL_V2_AMP_COEFFS,
)
from jaxwave.conversions import (
    as_array,
    component_masses_to_eta,
    tidal_combinations,
)
from jaxwave.approximants.imrphenomd import IMRPhenomD


# NRTidalv1 Padé coefficients (Dietrich et al. 2017 Eq. 7 / arXiv:1706.02969)
NRTIDAL_V1_COEFFS = {
    "n1": NRTIDAL_V1_PADÉ_COEFFS["n_1"],
    "n32": NRTIDAL_V1_PADÉ_COEFFS["n_3over2"],
    "n2": NRTIDAL_V1_PADÉ_COEFFS["n_2"],
    "n52": NRTIDAL_V1_PADÉ_COEFFS["n_5over2"],
    "d1": NRTIDAL_V1_PADÉ_COEFFS["d_1"],
    "d32": NRTIDAL_V1_PADÉ_COEFFS["d_3over2"],
}

# NRTidalv2 Padé coefficients (Dietrich et al. 2019 Eqs. 17-21 / arXiv:1905.06011)
N1_V2 = NRTIDAL_V2_PHASE_COEFFS["n_1"]
N32_V2 = NRTIDAL_V2_PHASE_COEFFS["n_3over2"]
N2_V2 = NRTIDAL_V2_PHASE_COEFFS["n_2"]
N52_V2 = NRTIDAL_V2_PHASE_COEFFS["n_5over2"]
N3_V2 = NRTIDAL_V2_PHASE_COEFFS["n_3"]
D1_V2 = NRTIDAL_V2_PHASE_COEFFS["d_1"]
D32_V2 = NRTIDAL_V2_PHASE_COEFFS["d_3over2"]
D2_V2 = NRTIDAL_V2_PHASE_COEFFS["d_2"]


def _nrtidalv2_paper_tidal_phase(x, eta, lambda_tilde):
    """Signed Dietrich et al. (2019) Eqs. (17)–(21) Fourier phase."""
    num = (1.0 + N1_V2 * x + N32_V2 * x**1.5 + N2_V2 * x**2
           + N52_V2 * x**2.5 + N3_V2 * x**3)
    den = 1.0 + D1_V2 * x + D32_V2 * x**1.5 + D2_V2 * x**2
    return -(117.0 / 256.0) * (lambda_tilde / eta) * x**2.5 * num / den


def _bns_merger_frequency(
    mass1: jnp.ndarray,
    mass2: jnp.ndarray,
    lambda1: jnp.ndarray,
    lambda2: jnp.ndarray,
) -> jnp.ndarray:
    """Compute BNS merger frequency f_merger in Hz from Dietrich et al. (2018) Eq. 11.

    Args:
        mass1, mass2: Component masses in solar masses (either order)
        lambda1, lambda2: Dimensionless quadrupolar tidal deformabilities

    Returns:
        f_mrg: Physical merger frequency in Hz
    """
    total_mass = mass1 + mass2
    # Ensure xa corresponds to the heavier component so mass ratio q = xb / xa <= 1
    m_high = jnp.maximum(mass1, mass2)
    m_low = jnp.minimum(mass1, mass2)
    xa = m_high / total_mass
    xb = m_low / total_mass
    m_sec = total_mass * MSUN_SEC

    # Effective tidal coupling parameter kappa_eff^T = (3/16) * Lambda_tilde
    lam_tilde, _ = tidal_combinations(mass1, mass2, lambda1, lambda2)
    kappa_eff_t = (3.0 / 16.0) * lam_tilde

    # Fit coefficients (Dietrich et al. 2018 Eq. 11)
    n1 = 3.354e-2
    n2 = 4.315e-5
    d1 = 7.542e-2
    d2 = 2.236e-4
    m_omega_0 = 0.3586

    ratio_sqrt = jnp.sqrt(jnp.maximum(xb / xa, 1e-8))
    num = 1.0 + n1 * kappa_eff_t + n2 * (kappa_eff_t ** 2.0)
    den = 1.0 + d1 * kappa_eff_t + d2 * (kappa_eff_t ** 2.0)
    m_omega_mrg = m_omega_0 * ratio_sqrt * (num / den)

    # Convert dimensionless GW angular frequency M*omega_mrg to frequency in Hz:
    # f_GW = omega_GW / (2 * pi) = (M * omega_mrg) / (2 * pi * M_sec)
    f_mrg = m_omega_mrg / (2.0 * PI * m_sec)
    return f_mrg


def _nrtidal_amplitude_shape(
    x: jnp.ndarray,
    lambda_tilde: jnp.ndarray,
) -> jnp.ndarray:
    """Dimensionless rational tidal-amplitude shape of Dietrich et al. Eq. (24)."""
    kappa_eff = (3.0 / 16.0) * lambda_tilde
    c = NRTIDAL_V2_AMP_COEFFS
    num = 1.0 + c["n_1"] * x + c["n_289"] * (x ** 2.89)
    den = 1.0 + c["d"] * (x ** 4.0)
    return -9.0 * kappa_eff * (x ** 3.25) * (num / den)


def _nrtidal_amplitude(
    x: jnp.ndarray,
    m_sec: jnp.ndarray,
    d_sec: jnp.ndarray,
    eta: jnp.ndarray,
    lambda_tilde: jnp.ndarray,
) -> jnp.ndarray:
    """Independent Pv2 tidal-amplitude component (Dietrich et al. 2019 Eq. 24).

    Table II excludes this component from the published D_NRTidalv2 waveform.

    A_T(f) = -9 * kappa_eff * x^(13/4) * (1 + n1 * x + n289 * x^2.89) / (1 + d * x^4)
    a_tidal = sqrt(5*pi*eta/24) * (m_sec^2 / d_sec) * A_T(f)
    """
    return (jnp.sqrt(5.0 * PI * eta / 24.0) * (m_sec ** 2.0 / d_sec)
            * _nrtidal_amplitude_shape(x, lambda_tilde))


def _binary_compatible_nrtidal_amplitude(
    x: jnp.ndarray,
    m_sec: jnp.ndarray,
    d_sec: jnp.ndarray,
    lambda_tilde: jnp.ndarray,
) -> jnp.ndarray:
    """Historical binary-verified 1/2 normalization; paper provenance unresolved."""
    return (m_sec ** 2.0 / (2.0 * d_sec)) * _nrtidal_amplitude_shape(x, lambda_tilde)


def _planck_taper_window(
    f: jnp.ndarray,
    f_start: jnp.ndarray,
    f_end: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate smooth Planck-taper window over [f_start, f_end].

    W(f) = 1 for f <= f_start
    W(f) = 0 for f >= f_end
    W(f) = 1 / (exp(-sigma(z)) + 1) where z = (f - f_start) / (f_end - f_start),
           sigma(z) = 1/z + 1/(z - 1)
    """
    z = (f - f_start) / (f_end - f_start + 1e-30)
    z_clamped = jnp.clip(z, 1e-7, 1.0 - 1e-7)
    sigma = (1.0 / z_clamped) + (1.0 / (z_clamped - 1.0))
    sigma_clamped = jnp.clip(sigma, -50.0, 50.0)
    w_smooth = 1.0 / (jnp.exp(-sigma_clamped) + 1.0)

    w = jnp.where(f <= f_start, 1.0, w_smooth)
    w = jnp.where(f >= f_end, 0.0, w)
    return jnp.where(f > 0.0, w, 0.0)


def _quadrupole_monopole_constants(
    lambda_val: jnp.ndarray,
    *,
    fit: str = "historical_2013",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return CQ and COc with an explicit quadrupole-fit choice.

    ``dietrich_2019`` uses printed Eqs. (28)–(29); the historical fit is kept
    for the binary-compatible default. Lambda clamps are separate native
    conventions and are outside the positive-Lambda paper-math gate.
    """
    lam_safe = jnp.maximum(lambda_val, 1.0)
    ln_lam = jnp.log(lam_safe)
    ln_lam2 = ln_lam * ln_lam
    ln_lam3 = ln_lam2 * ln_lam
    ln_lam4 = ln_lam3 * ln_lam

    if fit == "dietrich_2019":
        # Dietrich et al. (2019), Eq. (28), printed decimal precision.
        ln_cq = 0.1940 + 0.09163 * ln_lam + 0.04812 * ln_lam2 - 0.004286 * ln_lam3 + 0.00012450 * ln_lam4
    elif fit == "historical_2013":
        # Keep the inherited binary-compatible relation unchanged.
        ln_cq = 0.1940 + 0.0936 * ln_lam + 0.0474 * ln_lam2 - 0.00421 * ln_lam3 + 0.000123 * ln_lam4
    else:
        raise ValueError("fit must be 'historical_2013' or 'dietrich_2019'")
    c_q = jnp.exp(ln_cq)

    ln_cq_safe = jnp.maximum(ln_cq, 1e-8)
    ln_cq2 = ln_cq_safe * ln_cq_safe
    ln_cq3 = ln_cq2 * ln_cq_safe
    ln_cq4 = ln_cq3 * ln_cq_safe

    # log(C_Oc) = 0.003131 + 2.071 ln(C_Q) - 0.7152 ln(C_Q)^2 + 0.2458 ln(C_Q)^3 - 0.03309 ln(C_Q)^4
    ln_coc = 0.003131 + 2.071 * ln_cq_safe - 0.7152 * ln_cq2 + 0.2458 * ln_cq3 - 0.03309 * ln_cq4
    c_oc = jnp.exp(ln_coc)

    # For black holes (lambda = 0), CQ = COc = 1
    c_q = jnp.where(lambda_val > 1e-4, c_q, 1.0)
    c_oc = jnp.where(lambda_val > 1e-4, c_oc, 1.0)
    return c_q, c_oc


def _nrtidal_self_spin_components(
    x: jnp.ndarray,
    mass1: jnp.ndarray,
    mass2: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    lambda1: jnp.ndarray,
    lambda2: jnp.ndarray,
    *,
    fit: str = "historical_2013",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (S23, S35) self-spin phase corrections from Dietrich et al. (2019) Eqs. 26-27.

    Returns:
        s23: Sum of 2PN and 3PN self-spin phase contributions
        s35: 3.5PN self-spin phase contribution
    """
    total_mass = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    xa = mass1 / total_mass
    xb = mass2 / total_mass

    cq1, coc1 = _quadrupole_monopole_constants(lambda1, fit=fit)
    cq2, coc2 = _quadrupole_monopole_constants(lambda2, fit=fit)

    c_hat_q1 = cq1 - 1.0
    c_hat_q2 = cq2 - 1.0
    c_hat_oc1 = coc1 - 1.0
    c_hat_oc2 = coc2 - 1.0

    def star_terms(xa_val, xb_val, chia, chib, chat_q, chat_oc):
        xa2 = xa_val * xa_val
        xa3 = xa2 * xa_val
        chia2 = chia * chia
        chia3 = chia2 * chia

        # 2PN self-spin (Eq. 27a)
        psi_2pn = -50.0 * chat_q * xa2 * chia2

        # 3PN self-spin (Eq. 27b)
        psi_3pn = (5.0 / 84.0) * (9407.0 + 8218.0 * xa_val - 2016.0 * xa2) * chat_q * xa2 * chia2

        # 3.5PN self-spin (Eq. 27c)
        bracket_35 = (
            (xa2 + (308.0 / 3.0) * xa_val) * chia
            + ((xb_val * xb_val) - (89.0 / 3.0) * xb_val) * chib
            - 40.0 * PI
        )
        psi_35pn = 10.0 * bracket_35 * chat_q * xa2 * chia2 - 440.0 * chat_oc * xa3 * chia3
        return psi_2pn, psi_3pn, psi_35pn

    p2_1, p3_1, p35_1 = star_terms(xa, xb, chi1, chi2, c_hat_q1, c_hat_oc1)
    p2_2, p3_2, p35_2 = star_terms(xb, xa, chi2, chi1, c_hat_q2, c_hat_oc2)

    p2 = p2_1 + p2_2
    p3 = p3_1 + p3_2
    p35 = p35_1 + p35_2

    # Make dimensions broadcastable with x
    if x.ndim > p2.ndim:
        p2 = jnp.expand_dims(p2, -1)
        p3 = jnp.expand_dims(p3, -1)
        p35 = jnp.expand_dims(p35, -1)
        eta_u = jnp.expand_dims(eta, -1)
    else:
        eta_u = eta

    prefactor = 3.0 / (128.0 * eta_u * (x ** 2.5))
    s23 = prefactor * (p2 * (x ** 2.0) + p3 * (x ** 3.0))
    s35 = prefactor * (p35 * (x ** 3.5))
    return s23, s35


def assemble_nrtidalv2_pv2_aligned_carrier(
    f: jnp.ndarray, bbh_face_on: jnp.ndarray, mass1, mass2, chi1, chi2,
    lambda1, lambda2, distance_mpc,
) -> jnp.ndarray:
    """Compose published Pv2 BNS face-on carrier before precession rotation.

    Dietrich et al. (2019), Eqs. (3)–(4), (17)–(29), Table II, Sec. III.4.
    The supplied BBH carrier already has its phase/time origin selected.
    """
    f = jnp.asarray(f)
    bbh = jnp.asarray(bbh_face_on)
    def scalar(value):
        return jnp.asarray(value, dtype=f.dtype)
    m1, m2 = scalar(mass1), scalar(mass2)
    s1, s2 = scalar(chi1), scalar(chi2)
    lam1, lam2 = scalar(lambda1), scalar(lambda2)
    d_sec = scalar(distance_mpc) * MPC_SI / C_SI
    m_sec = (m1 + m2) * MSUN_SEC
    eta = component_masses_to_eta(m1, m2)
    lambda_tilde, _ = tidal_combinations(m1, m2, lam1, lam2)
    x = (PI * m_sec * f) ** (2.0 / 3.0)
    psi_t = _nrtidalv2_paper_tidal_phase(x, eta, lambda_tilde)
    s23, s35 = _nrtidal_self_spin_components(
        x, m1, m2, s1, s2, lam1, lam2, fit="dietrich_2019")
    a_t = _nrtidal_amplitude(x, m_sec, d_sec, eta, lambda_tilde)
    merger = _bns_merger_frequency(m1, m2, lam1, lam2)
    window = _planck_taper_window(f, merger, 1.2 * merger)
    modulus = jnp.abs(bbh)
    safe_modulus = jnp.where(modulus > 0, modulus, 1.0)
    unit = jnp.where(modulus > 0, bbh / safe_modulus, 0.0j)
    return window * (bbh + a_t * unit) * jnp.exp(-1j * (psi_t + s23 + s35))


def _nrtidal_self_spin_correction(
    x: jnp.ndarray,
    mass1: jnp.ndarray,
    mass2: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    lambda1: jnp.ndarray,
    lambda2: jnp.ndarray,
) -> jnp.ndarray:
    """Compute 2PN, 3PN, and 3.5PN self-spin / quadrupole-monopole phase corrections."""
    s23, s35 = _nrtidal_self_spin_components(
        x=x,
        mass1=mass1,
        mass2=mass2,
        chi1=chi1,
        chi2=chi2,
        lambda1=lambda1,
        lambda2=lambda2,
    )
    return s23 + s35


class IMRPhenomD_NRTidal:
    """IMRPhenomD_NRTidal Aligned-Spin Frequency-Domain BNS Waveform Model in JAX (Dietrich et al. 2017)."""

    def __init__(self, version: str = "v1", amplitude_variant: str = "binary_compatible"):
        self.version = version.lower()
        if amplitude_variant not in ("binary_compatible", "paper_2019"):
            raise ValueError("amplitude_variant must be 'binary_compatible' or 'paper_2019'")
        self.amplitude_variant = amplitude_variant

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("version", self.version)
        kwargs.setdefault("amplitude_variant", self.amplitude_variant)
        return self.evaluate(*args, **kwargs)

    @classmethod
    def evaluate(
        cls,
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        lambda1: Optional[Union[float, jnp.ndarray]] = None,
        lambda2: Optional[Union[float, jnp.ndarray]] = None,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        apply_merger_taper: bool = True,
        return_phase: bool = False,
        version: str = "v1",
        amplitude_variant: str = "binary_compatible",
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Generate IMRPhenomD_NRTidal frequency-domain polarizations (h_plus, h_cross).

        Args:
            f: Frequency array in Hz
            mass1, mass2: Component masses in solar masses (either order)
            chi1, chi2: Aligned dimensionless spins
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            lambda1, lambda2: Dimensionless tidal deformabilities
            f_ref: Reference frequency in Hz
            phasing_convention: 'coalescence' or 'reference_frequency'
            reference_frame: Phasing frame convention
            apply_merger_taper: Whether to apply the Planck taper window at f_merger
            return_phase: Whether to return the total phase
            version: 'v1' or 'v2'
            amplitude_variant: 'binary_compatible' (default) or 'paper_2019'

        Returns:
            (h_plus, h_cross) or (h_plus, h_cross, total_phase)
        """
        if amplitude_variant not in ("binary_compatible", "paper_2019"):
            raise ValueError("amplitude_variant must be 'binary_compatible' or 'paper_2019'")
        dtype = jnp.float64
        m1 = as_array(mass1, dtype=dtype)
        m2 = as_array(mass2, dtype=dtype)
        s1 = as_array(chi1, dtype=dtype)
        s2 = as_array(chi2, dtype=dtype)
        lam1 = as_array(0.0 if lambda1 is None else lambda1, dtype=dtype)
        lam2 = as_array(0.0 if lambda2 is None else lambda2, dtype=dtype)
        f_arr = as_array(f, dtype=dtype)

        # 1. Base point-particle IMRPhenomD polarizations
        bbh_res = IMRPhenomD.evaluate(
            f=f_arr,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=distance,
            inclination=inclination,
            phic=phic,
            tc=tc,
            f_ref=f_ref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            return_phase=return_phase,
        )
        if return_phase:
            hp_bbh, hc_bbh, bbh_phase = bbh_res
        else:
            hp_bbh, hc_bbh = bbh_res

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        m_sec = total_mass * MSUN_SEC

        is_batched = total_mass.ndim >= 1
        if f_arr.ndim == 1 and is_batched:
            f_eval = jnp.expand_dims(f_arr, 0)
            m_sec_eval = jnp.expand_dims(m_sec, -1)
            eta_eval = jnp.expand_dims(eta, -1)
        else:
            f_eval = f_arr
            m_sec_eval = m_sec
            eta_eval = eta

        # PN expansion parameter x = (pi * M * f)^(2/3)
        safe_f = jnp.maximum(f_eval, 1e-8)
        x = (PI * m_sec_eval * safe_f) ** (2.0 / 3.0)

        # Tidal parameters: Lambda_tilde and leading factor
        lam_tilde, _ = tidal_combinations(m1, m2, lam1, lam2)
        if is_batched and f_eval.ndim > lam_tilde.ndim:
            lam_tilde_eval = jnp.expand_dims(lam_tilde, -1)
        else:
            lam_tilde_eval = lam_tilde

        leading_factor = -(117.0 / 256.0) * (lam_tilde_eval / eta_eval)

        # 2. Tidal and self-spin phase corrections
        ver = version.lower()
        if ver == "v1":
            c = NRTIDAL_V1_COEFFS
            num = (
                1.0
                + c["n1"] * x
                + c["n32"] * (x ** 1.5)
                + c["n2"] * (x ** 2.0)
                + c["n52"] * (x ** 2.5)
            )
            den = 1.0 + c["d1"] * x + c["d32"] * (x ** 1.5)
            pade_ratio = num / den
            psi_tidal = leading_factor * (x ** 2.5) * pade_ratio
            total_delta_psi = psi_tidal
        else:
            num = (
                1.0
                + N1_V2 * x
                + N32_V2 * (x ** 1.5)
                + N2_V2 * (x ** 2.0)
                + N52_V2 * (x ** 2.5)
                + N3_V2 * (x ** 3.0)
            )
            den = 1.0 + D1_V2 * x + D32_V2 * (x ** 1.5) + D2_V2 * (x ** 2.0)
            pade_ratio = num / den
            psi_tidal = leading_factor * (x ** 2.5) * pade_ratio

            self_spin_fit = "dietrich_2019" if amplitude_variant == "paper_2019" else "historical_2013"
            s23, s35 = _nrtidal_self_spin_components(
                x=x,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                lambda1=lam1,
                lambda2=lam2,
                fit=self_spin_fit,
            )

            conv = reference_frame if reference_frame is not None else (
                phasing_convention if phasing_convention is not None else ("reference_frequency" if f_ref is not None else "coalescence")
            )
            conv = conv.lower()

            if conv in ("reference_frequency", "reffreq", "ref"):
                ref_f = f_ref if f_ref is not None else (f_eval.reshape(-1)[0] if f_eval.size > 0 else 20.0)
                fref_t = as_array(ref_f, dtype=dtype)
                if is_batched and f_eval.ndim > fref_t.ndim:
                    fref_u = jnp.expand_dims(fref_t, -1)
                else:
                    fref_u = fref_t

                safe_fref = jnp.maximum(fref_u, 1e-8)
                x_ref = (PI * m_sec_eval * safe_fref) ** (2.0 / 3.0)
                s23_ref, _ = _nrtidal_self_spin_components(
                    x=x_ref,
                    mass1=m1,
                    mass2=m2,
                    chi1=s1,
                    chi2=s2,
                    lambda1=lam1,
                    lambda2=lam2,
                    fit=self_spin_fit,
                )
                total_delta_psi = psi_tidal + (s23 - s23_ref)
                if amplitude_variant == "binary_compatible":
                    total_delta_psi = total_delta_psi + s35
            else:
                total_delta_psi = psi_tidal + s23
                if amplitude_variant == "binary_compatible":
                    total_delta_psi = total_delta_psi + s35

        # 3. High-frequency BNS merger cutoff & Planck taper
        f_merger = _bns_merger_frequency(m1, m2, lam1, lam2)
        if is_batched and f_eval.ndim > f_merger.ndim:
            f_merger = jnp.expand_dims(f_merger, -1)
        w_taper_calc = _planck_taper_window(f_eval, f_merger, 1.2 * f_merger)

        if isinstance(apply_merger_taper, (bool, int)):
            w_taper = w_taper_calc if apply_merger_taper else jnp.ones_like(f_eval)
        else:
            w_taper = jnp.where(apply_merger_taper, w_taper_calc, 1.0)

        # Complex phase factor exp(-i * delta_psi)
        exp_factor = jnp.cos(total_delta_psi) - 1j * jnp.sin(total_delta_psi)

        amplitude_factor = jnp.ones_like(f_eval)
        if ver == "v2" and amplitude_variant == "binary_compatible":
            # Preserve the historical native default, validated against the
            # pinned binary. Its 1/2 amplitude normalization is not Eq. (24).
            dist_t = as_array(distance, dtype=dtype)
            d_sec = (dist_t * MPC_SI) / C_SI
            if is_batched and f_eval.ndim > d_sec.ndim:
                d_sec = jnp.expand_dims(d_sec, -1)
            a_tidal = _binary_compatible_nrtidal_amplitude(
                x, m_sec_eval, d_sec, lam_tilde_eval)
            a_tidal = jnp.where(f_eval > 0.0, a_tidal, 0.0)
            inc = as_array(inclination, dtype=dtype)
            if is_batched and f_eval.ndim > inc.ndim:
                inc = jnp.expand_dims(inc, -1)
            plus_projection = 0.5 * (1.0 + jnp.cos(inc)**2)
            safe_hp = jnp.where(f_eval > 0.0, hp_bbh, 1.0 + 0.0j)
            base_amplitude = jnp.abs(safe_hp) / plus_projection
            positive = (base_amplitude > 0.0) & (f_eval > 0.0)
            safe_amplitude = jnp.where(positive, base_amplitude, 1.0)
            amplitude_factor = jnp.where(
                positive, 1.0 + a_tidal / safe_amplitude, 1.0)
        # Dietrich et al. (2019), Table II gives the paper_2019 D option no
        # tidal-amplitude correction. Eq. (24) is checked independently.
        h_plus = hp_bbh * amplitude_factor * exp_factor * w_taper
        h_cross = hc_bbh * amplitude_factor * exp_factor * w_taper

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus = jnp.where(f_eval > 0.0, h_plus, 0.0 + 0.0j)
        h_cross = jnp.where(f_eval > 0.0, h_cross, 0.0 + 0.0j)

        if return_phase:
            return h_plus, h_cross, bbh_phase - total_delta_psi

        return h_plus, h_cross


class IMRPhenomD_NRTidalv2(IMRPhenomD_NRTidal):
    """NRTidalv2 D model: historical binary-compatible default or paper_2019 D."""

    def __init__(self, amplitude_variant: str = "binary_compatible"):
        super().__init__(version="v2", amplitude_variant=amplitude_variant)

    @classmethod
    def evaluate(cls, *args, **kwargs):
        kwargs.setdefault("version", "v2")
        return super().evaluate(*args, **kwargs)
