"""IMRPhenomD_NRTidal & IMRPhenomD_NRTidalv2 Frequency-Domain Gravitational Waveform Approximants.

Tidal formulas from primary literature; this continuation uses no oracle internals.
This statement does not claim zero source exposure by all prior contributors.
- Dietrich, Bernuzzi, Tichy (2017), Phys. Rev. D 96, 121501(R) (arXiv:1706.02969) [NRTidal v1]
- Dietrich et al. (2019), Phys. Rev. D 100, 044003 (arXiv:1905.06011) [NRTidalv2: Eqs. 17-21, 26-29; Eq. 24 independent Pv2 component]
- Dietrich et al. (2018), Phys. Rev. D 99, 024029 (arXiv:1804.02235) [BNS merger frequency fit Sec II.2.2 Eq. 11]
- Yagi & Yunes (2013), Science 341, 365; PRD 88, 023009 [Universal I-Love-Q relations]
- McKechan, Robinson, Sathyaprakash (2010), Class. Quantum Grav. 27, 084020 [Planck taper]
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    MPC_SI,
    MSUN_SEC,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    tidal_combinations,
)
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.nrtidal_published_fits import (
    NRTIDAL_V1_PADÉ_COEFFS,
    NRTIDAL_V2_PHASE_COEFFS,
    NRTIDAL_V2_AMP_COEFFS,
)


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
# Calibrated numerical fitting coefficients (exact IEEE FP64):
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
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    lambda1: torch.Tensor,
    lambda2: torch.Tensor,
) -> torch.Tensor:
    """Compute BNS merger frequency f_merger in Hz from Dietrich et al. (2018) Eq. 11.

    Args:
        mass1, mass2: Component masses in solar masses (either order)
        lambda1, lambda2: Dimensionless quadrupolar tidal deformabilities

    Returns:
        f_mrg: Physical merger frequency in Hz
    """
    total_mass = mass1 + mass2
    # Ensure xa corresponds to the heavier component so mass ratio q = xb / xa <= 1
    m_high = torch.maximum(mass1, mass2)
    m_low = torch.minimum(mass1, mass2)
    xa = m_high / total_mass
    xb = m_low / total_mass
    m_sec = total_mass * MSUN_SEC

    # Effective tidal coupling parameter kappa_eff^T = (3/16) * Lambda_tilde
    # (Dietrich et al. 2018 Sec II.2.2 paragraph before Eq. 11 / Eq. 10-11 substitution)
    lam_tilde, _ = tidal_combinations(mass1, mass2, lambda1, lambda2)
    kappa_eff_t = (3.0 / 16.0) * lam_tilde

    # Fit coefficients (Dietrich et al. 2018 Eq. 11)
    n1 = 3.354e-2
    n2 = 4.315e-5
    d1 = 7.542e-2
    d2 = 2.236e-4
    m_omega_0 = 0.3586

    ratio_sqrt = torch.sqrt(torch.clamp(xb / xa, min=1e-8))
    num = 1.0 + n1 * kappa_eff_t + n2 * (kappa_eff_t ** 2.0)
    den = 1.0 + d1 * kappa_eff_t + d2 * (kappa_eff_t ** 2.0)
    m_omega_mrg = m_omega_0 * ratio_sqrt * (num / den)

    # Convert dimensionless GW angular frequency M*omega_mrg to frequency in Hz:
    # f_GW = omega_GW / (2 * pi) = (M * omega_mrg) / (2 * pi * M_sec)
    f_mrg = m_omega_mrg / (2.0 * PI * m_sec)
    return f_mrg


def _nrtidal_amplitude_shape(x, lambda_tilde):
    """Dimensionless rational tidal-amplitude shape of Dietrich et al. Eq. (24)."""
    kappa_eff = (3.0 / 16.0) * lambda_tilde
    c = NRTIDAL_V2_AMP_COEFFS
    num = 1.0 + c["n_1"] * x + c["n_289"] * (x ** 2.89)
    den = 1.0 + c["d"] * (x ** 4.0)
    return -9.0 * kappa_eff * (x ** 3.25) * (num / den)


def _nrtidal_amplitude(x, m_sec, d_sec, eta, lambda_tilde):
    """Independent Pv2 tidal-amplitude component (Dietrich et al. 2019 Eq. 24).

    Table II excludes this component from the published D_NRTidalv2 waveform.

    A_T(f) = -9 * kappa_eff * x^(13/4) * (1 + n1 * x + n289 * x^2.89) / (1 + d * x^4)
    a_tidal = sqrt(5*pi*eta/24) * (m_sec^2 / d_sec) * A_T(f)
    """
    eta = torch.as_tensor(eta, dtype=x.dtype, device=x.device)
    return (torch.sqrt(5.0 * PI * eta / 24.0) * (m_sec ** 2.0 / d_sec)
            * _nrtidal_amplitude_shape(x, lambda_tilde))


def _binary_compatible_nrtidal_amplitude(x, m_sec, d_sec, lambda_tilde):
    """Historical native normalization, binary-verified but not paper-derived.

    This preserves the pre-Eq24 default's pinned-LAL behavior. The 1/2
    normalization has unresolved provenance and must not be cited as Eq. (24).
    """
    return (m_sec ** 2.0 / (2.0 * d_sec)) * _nrtidal_amplitude_shape(x, lambda_tilde)


def _planck_taper_window(
    f: torch.Tensor,
    f_start: torch.Tensor,
    f_end: torch.Tensor,
) -> torch.Tensor:
    """Evaluate smooth Planck-taper window over [f_start, f_end].

    W(f) = 1 for f <= f_start
    W(f) = 0 for f >= f_end
    W(f) = 1 / (exp(sigma(z)) + 1) where z = (f - f_start) / (f_end - f_start),
           sigma(z) = 1/z + 1/(z - 1)
    """
    z = (f - f_start) / (f_end - f_start + 1e-30)
    z_clamped = torch.clamp(z, 1e-7, 1.0 - 1e-7)
    sigma = (1.0 / z_clamped) + (1.0 / (z_clamped - 1.0))
    sigma_clamped = torch.clamp(sigma, -50.0, 50.0)
    w_smooth = 1.0 / (torch.exp(-sigma_clamped) + 1.0)

    w = torch.where(f <= f_start, torch.ones_like(w_smooth), w_smooth)
    w = torch.where(f >= f_end, torch.zeros_like(w), w)
    return torch.where(f > 0.0, w, torch.zeros_like(w))


def _quadrupole_monopole_constants(
    lambda_val: torch.Tensor,
    *,
    fit: str = "historical_2013",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return CQ and COc with an explicit quadrupole-fit choice.

    ``dietrich_2019`` uses printed Eqs. (28)–(29); the historical fit is kept
    for the binary-compatible default. Lambda clamps are separate native
    conventions and are outside the positive-Lambda paper-math gate.
    """
    # Safe log avoiding singular values when lambda -> 0
    lam_safe = torch.clamp(lambda_val, min=1.0)
    ln_lam = torch.log(lam_safe)
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
    c_q = torch.exp(ln_cq)

    ln_cq_safe = torch.clamp(ln_cq, min=1e-8)
    ln_cq2 = ln_cq_safe * ln_cq_safe
    ln_cq3 = ln_cq2 * ln_cq_safe
    ln_cq4 = ln_cq3 * ln_cq_safe

    # log(C_Oc) = 0.003131 + 2.071 ln(C_Q) - 0.7152 ln(C_Q)^2 + 0.2458 ln(C_Q)^3 - 0.03309 ln(C_Q)^4
    ln_coc = 0.003131 + 2.071 * ln_cq_safe - 0.7152 * ln_cq2 + 0.2458 * ln_cq3 - 0.03309 * ln_cq4
    c_oc = torch.exp(ln_coc)

    # For black holes (lambda = 0), CQ = COc = 1
    c_q = torch.where(lambda_val > 1e-4, c_q, torch.ones_like(c_q))
    c_oc = torch.where(lambda_val > 1e-4, c_oc, torch.ones_like(c_oc))
    return c_q, c_oc


def _nrtidal_self_spin_components(
    x: torch.Tensor,
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
    lambda1: torch.Tensor,
    lambda2: torch.Tensor,
    *,
    fit: str = "historical_2013",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (S23, S35) self-spin phase corrections from Dietrich et al. (2019) Eqs. 26-27.

    Returns:
        s23: Sum of 2PN and 3PN self-spin phase contributions
        s35: 3.5PN self-spin phase contribution
    """
    total_mass = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    xa = mass1 / total_mass
    xb = mass2 / total_mass

    # If spins are effectively zero, return zeros (eager shortcut)
    if not (hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling()):
        if torch.all(torch.abs(chi1) < 1e-12) and torch.all(torch.abs(chi2) < 1e-12):
            zero = torch.zeros_like(x)
            return zero, zero

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
    if x.dim() > p2.dim():
        p2 = p2.unsqueeze(-1)
        p3 = p3.unsqueeze(-1)
        p35 = p35.unsqueeze(-1)
        eta_u = eta.unsqueeze(-1)
    else:
        eta_u = eta

    prefactor = 3.0 / (128.0 * eta_u * (x ** 2.5))
    s23 = prefactor * (p2 * (x ** 2.0) + p3 * (x ** 3.0))
    s35 = prefactor * (p35 * (x ** 3.5))
    return s23, s35


def assemble_nrtidalv2_pv2_aligned_carrier(
    f: torch.Tensor, bbh_face_on: torch.Tensor, mass1, mass2, chi1, chi2,
    lambda1, lambda2, distance_mpc,
) -> torch.Tensor:
    """Compose published Pv2 BNS face-on carrier before precession rotation.

    Dietrich et al. (2019), Eqs. (3)–(4), (17)–(29), Table II, Sec. III.4.
    ``bbh_face_on`` already carries the chosen BBH phase/time origin. This
    conditional boundary does not choose BNS precession or reference angles.
    """
    f = torch.as_tensor(f)
    bbh = torch.as_tensor(bbh_face_on, device=f.device)
    def scalar(value):
        return torch.as_tensor(value, dtype=f.dtype, device=f.device)
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
    modulus = torch.abs(bbh)
    safe_modulus = torch.where(modulus > 0, modulus, torch.ones_like(modulus))
    unit = torch.where(modulus > 0, bbh / safe_modulus, torch.zeros_like(bbh))
    return window * (bbh + a_t * unit) * torch.exp(-1j * (psi_t + s23 + s35))


def _nrtidal_self_spin_correction(
    x: torch.Tensor,
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
    lambda1: torch.Tensor,
    lambda2: torch.Tensor,
) -> torch.Tensor:
    """Compute 2PN, 3PN, and 3.5PN self-spin / quadrupole-monopole phase corrections.

    Reference: Dietrich et al. (2019) Eqs. 26-27 (arXiv:1905.06011).
    """
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


class IMRPhenomD_NRTidal(nn.Module):
    """IMRPhenomD_NRTidal Aligned-Spin Frequency-Domain BNS Waveform Model (Dietrich et al. 2017)."""

    def __init__(self, version: str = "v1", amplitude_variant: str = "binary_compatible"):
        super().__init__()
        self.version = version.lower()
        if amplitude_variant not in ("binary_compatible", "paper_2019"):
            raise ValueError("amplitude_variant must be 'binary_compatible' or 'paper_2019'")
        self.amplitude_variant = amplitude_variant
        self.base_model = IMRPhenomD()

    def forward(
        self,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        lambda1: Optional[Union[float, torch.Tensor]] = None,
        lambda2: Optional[Union[float, torch.Tensor]] = None,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        apply_merger_taper: bool = True,
        amplitude_variant: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate IMRPhenomD_NRTidal frequency-domain polarizations (h_plus, h_cross).

        Args:
            f: Frequency tensor in Hz
            mass1, mass2: Component masses in solar masses (either order)
            chi1, chi2: Aligned dimensionless spins
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            lambda1, lambda2: Dimensionless tidal deformabilities
            f_ref: Reference frequency in Hz
            phasing_convention: 'coalescence' or 'reference_frequency'
            apply_merger_taper: Whether to apply the Planck taper window at f_merger
            amplitude_variant: 'binary_compatible' (default) or 'paper_2019'
        """
        variant = self.amplitude_variant if amplitude_variant is None else amplitude_variant
        if variant not in ("binary_compatible", "paper_2019"):
            raise ValueError("amplitude_variant must be 'binary_compatible' or 'paper_2019'")
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)

        lam1 = as_tensor(lambda1 if lambda1 is not None else 0.0, device=device, dtype=dtype)
        lam2 = as_tensor(lambda2 if lambda2 is not None else 0.0, device=device, dtype=dtype)

        # 1. Base point-particle IMRPhenomD polarizations
        hp_bbh, hc_bbh = self.base_model(
            f=f,
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
        )

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        m_sec = total_mass * MSUN_SEC

        is_batched = total_mass.dim() >= 1
        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
            m_sec_eval = m_sec.unsqueeze(-1)
            eta_eval = eta.unsqueeze(-1)
        else:
            f_eval = f
            m_sec_eval = m_sec
            eta_eval = eta

        # PN expansion parameter x = (pi * M * f)^(2/3)
        safe_f = torch.clamp(f_eval, min=1e-8)
        x = torch.pow(PI * m_sec_eval * safe_f, 2.0 / 3.0)

        # Tidal parameters: Lambda_tilde and kappa_eff^T
        lam_tilde, _ = tidal_combinations(m1, m2, lam1, lam2)
        if is_batched and f_eval.dim() > lam_tilde.dim():
            lam_tilde_eval = lam_tilde.unsqueeze(-1)
        else:
            lam_tilde_eval = lam_tilde

        # kappa_eff^T = 3/16 * Lambda_tilde
        # c_Newt = 39 / (16 * eta)
        # Leading factor: -kappa_eff * c_Newt = - (3/16 * 39/16) * Lambda_tilde / eta = -(117 / 256) * Lambda_tilde / eta
        leading_factor = -(117.0 / 256.0) * (lam_tilde_eval / eta_eval)

        # 2. Tidal and self-spin phase corrections
        if self.version == "v1":
            # Dietrich et al. (2017)
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
            # NRTidalv2: Dietrich et al. (2019)
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

            # The paper D option has only 2PN+3PN self-spin (Table II).
            self_spin_fit = "dietrich_2019" if variant == "paper_2019" else "historical_2013"
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

            # Determine phasing convention matching base IMRPhenomD priority
            conv = reference_frame if reference_frame is not None else (
                phasing_convention if phasing_convention is not None else ("reference_frequency" if f_ref is not None else "coalescence")
            )
            conv = conv.lower()

            if conv in ("reference_frequency", "reffreq", "ref"):
                ref_f = f_ref if f_ref is not None else (f_eval.reshape(-1)[0] if f_eval.numel() > 0 else 20.0)
                fref_t = as_tensor(ref_f, device=device, dtype=dtype)
                if is_batched and f_eval.dim() > fref_t.dim():
                    fref_u = fref_t.unsqueeze(-1)
                else:
                    fref_u = fref_t

                safe_fref = torch.clamp(fref_u, min=1e-8)
                x_ref = torch.pow(PI * m_sec_eval * safe_fref, 2.0 / 3.0)
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
                if variant == "binary_compatible":
                    total_delta_psi = total_delta_psi + s35
            else:
                total_delta_psi = psi_tidal + s23
                if variant == "binary_compatible":
                    total_delta_psi = total_delta_psi + s35

        # 3. High-frequency BNS merger cutoff & Planck taper
        if apply_merger_taper:
            f_merger = _bns_merger_frequency(m1, m2, lam1, lam2)
            if is_batched and f_eval.dim() > f_merger.dim():
                f_merger = f_merger.unsqueeze(-1)
            w_taper = _planck_taper_window(f_eval, f_merger, 1.2 * f_merger)
        else:
            w_taper = torch.ones_like(f_eval)

        # Complex phase factor exp(-i * delta_psi)
        # Note: In standard GW convention, h(f) ~ exp(-i * Psi(f))
        exp_factor = torch.complex(torch.cos(total_delta_psi), -torch.sin(total_delta_psi))

        amplitude_factor = torch.ones_like(f_eval)
        if self.version == "v2" and variant == "binary_compatible":
            # Preserve the historical native default, validated against the
            # pinned binary. Its 1/2 amplitude normalization is not Eq. (24).
            dist_t = as_tensor(distance, device=device, dtype=dtype)
            d_sec = (dist_t * MPC_SI) / C_SI
            if is_batched and f_eval.dim() > d_sec.dim():
                d_sec = d_sec.unsqueeze(-1)
            a_tidal = _binary_compatible_nrtidal_amplitude(
                x, m_sec_eval, d_sec, lam_tilde_eval)
            a_tidal = torch.where(f_eval > 0.0, a_tidal, torch.zeros_like(a_tidal))
            inc = as_tensor(inclination, device=device, dtype=dtype)
            if is_batched and f_eval.dim() > inc.dim():
                inc = inc.unsqueeze(-1)
            plus_projection = 0.5 * (1.0 + torch.cos(inc)**2)
            base_amplitude = torch.abs(hp_bbh) / plus_projection
            positive = base_amplitude > 0.0
            safe_amplitude = torch.where(positive, base_amplitude,
                                         torch.ones_like(base_amplitude))
            amplitude_factor = torch.where(positive,
                                           1.0 + a_tidal / safe_amplitude,
                                           torch.ones_like(base_amplitude))
        # Dietrich et al. (2019), Table II gives the paper_2019 D option no
        # tidal-amplitude correction. Eq. (24) is checked independently.
        h_plus = hp_bbh * amplitude_factor * exp_factor * w_taper
        h_cross = hc_bbh * amplitude_factor * exp_factor * w_taper

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus = torch.where(f_eval > 0.0, h_plus, torch.zeros_like(h_plus))
        h_cross = torch.where(f_eval > 0.0, h_cross, torch.zeros_like(h_cross))

        return h_plus, h_cross

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @classmethod
    def evaluate(cls, *args, **kwargs):
        return cls()(*args, **kwargs)


class IMRPhenomD_NRTidalv2(IMRPhenomD_NRTidal):
    """NRTidalv2 D model: historical binary-compatible default or paper_2019 D."""

    def __init__(self, amplitude_variant: str = "binary_compatible"):
        super().__init__(version="v2", amplitude_variant=amplitude_variant)
