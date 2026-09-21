"""Approximants package for jaxwave."""

from .taylorf2 import TaylorF2
from .imrphenomd import IMRPhenomD
from .imrphenomt import IMRPhenomT
from .imrphenomd_nrtidal import IMRPhenomD_NRTidal, IMRPhenomD_NRTidalv2
from .taylorf2_hm import TaylorF2_HM
from .imrphenom_hm import IMRPhenomHM
from .imrphenomp import IMRPhenomP
from .phenompv2_waveform import IMRPhenomPv2
from .taylorf2_eccentric import TaylorF2_Eccentric
from .imrphenom_xas import IMRPhenomXAS
from .imrphenom_xhm import IMRPhenomXHM
from .imrphenom_nsbh import IMRPhenomNSBH
from .kerr_ringdown import (
    KerrRingdown,
    get_ringdown_waveform,
    sampled_kerr_single_mode_reference,
    kerr_single_mode_reference,
    kerr_single_mode_fourier_reference,
    ringdown_amplitude_from_energy_fraction,
    evaluate_qnm,
)
from .continuous_wave import (
    ContinuousWave,
    continuous_wave_strain,
    continuous_wave_h0,
    continuous_wave_basis,
    continuous_wave_amplitudes,
    continuous_wave_phase,
)
from .emri_kludge import (
    EMRIKludge,
    kerr_geodesic_frequencies,
    peters_mathews_radiation_reaction,
)
from .ppe import (
    ParameterizedPostEinstein,
    ppe_phase_shift,
    modified_dispersion_phase_shift,
    PPE_PN_MAPPING,
)
from .precession_ode import SpinPrecessionODE
from .imrphenom_xp import IMRPhenomXP, wigner_small_d_matrix
from .precession_msa import (
    elliptic_f,
    jacobi_sn_cn_dn,
    compute_effective_chip,
    compute_l_3pn,
    compute_msa_final_spin,
    evaluate_msa_precession,
    resolve_precession_version,
)
from .imrphenom_xphm import IMRPhenomXPHM
from .precession_spintaylor import (
    compute_chi_tot_perp,
    SpinTaylorT4Engine,
    evaluate_spintaylor_precession,
)
from .pnr_coefficients import COEFF_ARRAYS as PNR_COEFF_ARRAYS
from .precession_pnr import (
    compute_pnr_single_spin,
    compute_pnr_alpha_coefficients,
    compute_pnr_beta_coefficients,
    eval_pnr_alpha_mr,
    eval_pnr_beta_mr,
    pnr_angles_window,
    eval_pnr_linear_frequency_map,
    evaluate_pnr_precession,
)
from .eob.waveform import SEOBNR_ODE
from .eob.published_waveform import (
    SEOBNRv4Published,
    SEOBNRv4PublishedNonspinning,
)
from .teobresums import (
    TEOBResumS,
    TEOBResumSHamiltonian,
    TEOBResumSRadiationReaction,
    compute_tidal_merger_frequency,
    _piecewise_linear_fourier,
    _teobresums_orbital_potential,
    _teobresums_centrifugal_radius,
    _teobresums_effective_hamiltonian,
)
from .memory import (
    MemoryWaveform,
    compute_memory_h20,
    memory_polarizations,
    add_memory_to_waveform,
)
from .phenom_meco import (
    hybrid_binding_energy,
    hybrid_meco_velocity,
    hybrid_meco_frequency,
)

__all__ = [
    "SEOBNR_ODE",
    "SEOBNRv4Published",
    "SEOBNRv4PublishedNonspinning",
    "TaylorF2",
    "IMRPhenomD",
    "IMRPhenomT",
    "IMRPhenomD_NRTidal",
    "IMRPhenomD_NRTidalv2",
    "TaylorF2_HM",
    "IMRPhenomHM",
    "IMRPhenomP",
    "IMRPhenomPv2",
    "TaylorF2_Eccentric",
    "IMRPhenomXAS",
    "IMRPhenomXHM",
    "IMRPhenomNSBH",
    "KerrRingdown",
    "get_ringdown_waveform",
    "sampled_kerr_single_mode_reference",
    "kerr_single_mode_reference",
    "kerr_single_mode_fourier_reference",
    "ringdown_amplitude_from_energy_fraction",
    "evaluate_qnm",
    "ContinuousWave",
    "continuous_wave_strain",
    "continuous_wave_h0",
    "continuous_wave_basis",
    "continuous_wave_amplitudes",
    "continuous_wave_phase",
    "EMRIKludge",
    "kerr_geodesic_frequencies",
    "peters_mathews_radiation_reaction",
    "ParameterizedPostEinstein",
    "ppe_phase_shift",
    "modified_dispersion_phase_shift",
    "PPE_PN_MAPPING",
    "SpinPrecessionODE",
    "IMRPhenomXP",
    "IMRPhenomXPHM",
    "wigner_small_d_matrix",
    "elliptic_f",
    "jacobi_sn_cn_dn",
    "compute_effective_chip",
    "compute_l_3pn",
    "compute_msa_final_spin",
    "evaluate_msa_precession",
    "resolve_precession_version",
    "compute_chi_tot_perp",
    "SpinTaylorT4Engine",
    "evaluate_spintaylor_precession",
    "PNR_COEFF_ARRAYS",
    "compute_pnr_single_spin",
    "compute_pnr_alpha_coefficients",
    "compute_pnr_beta_coefficients",
    "eval_pnr_alpha_mr",
    "eval_pnr_beta_mr",
    "pnr_angles_window",
    "eval_pnr_linear_frequency_map",
    "evaluate_pnr_precession",
    "TEOBResumS",
    "TEOBResumSHamiltonian",
    "TEOBResumSRadiationReaction",
    "compute_tidal_merger_frequency",
    "_piecewise_linear_fourier",
    "_teobresums_orbital_potential",
    "_teobresums_centrifugal_radius",
    "_teobresums_effective_hamiltonian",
    "MemoryWaveform",
    "compute_memory_h20",
    "memory_polarizations",
    "add_memory_to_waveform",
    "hybrid_binding_energy",
    "hybrid_meco_velocity",
    "hybrid_meco_frequency",
]



