"""jaxwave: Batched, differentiable gravitational-wave waveform generator in JAX."""

import jax

# Automatically enable float64 in JAX for gravitational-wave phase precision
jax.config.update("jax_enable_x64", True)

from .waveforms import get_fd_waveform, get_td_waveform, WaveformGenerator
from .approximants.memory import (
    MemoryWaveform,
    compute_memory_h20,
    memory_polarizations,
    add_memory_to_waveform,
)
from .approximants.phenom_meco import (
    hybrid_binding_energy,
    hybrid_meco_velocity,
    hybrid_meco_frequency,
)
from .approximants.taylorf2 import TaylorF2
from .approximants.imrphenomd import IMRPhenomD
from .approximants.imrphenomt import IMRPhenomT
from .approximants.imrphenomd_nrtidal import IMRPhenomD_NRTidal, IMRPhenomD_NRTidalv2
from .approximants.taylorf2_hm import TaylorF2_HM
from .approximants.imrphenom_hm import IMRPhenomHM
from .approximants.imrphenomp import IMRPhenomP
from .approximants.phenompv2_waveform import IMRPhenomPv2
from .approximants.taylorf2_eccentric import TaylorF2_Eccentric
from .approximants.imrphenom_xas import IMRPhenomXAS
from .approximants.imrphenom_xp import IMRPhenomXP
from .approximants.imrphenom_xphm import IMRPhenomXPHM
from .approximants.imrphenom_xhm import IMRPhenomXHM
from .approximants.imrphenom_nsbh import IMRPhenomNSBH
from .approximants.kerr_ringdown import (
    KerrRingdown,
    get_ringdown_waveform,
    sampled_kerr_single_mode_reference,
    kerr_single_mode_reference,
    kerr_single_mode_fourier_reference,
    ringdown_amplitude_from_energy_fraction,
    evaluate_qnm,
)
from .approximants.continuous_wave import (
    ContinuousWave,
    continuous_wave_strain,
    continuous_wave_h0,
    continuous_wave_basis,
    continuous_wave_amplitudes,
    continuous_wave_phase,
)
from .approximants.emri_kludge import (
    EMRIKludge,
    kerr_geodesic_frequencies,
    peters_mathews_radiation_reaction,
)
from .approximants.ppe import (
    ParameterizedPostEinstein,
    ppe_phase_shift,
    modified_dispersion_phase_shift,
    PPE_PN_MAPPING,
)
from .approximants.precession_ode import SpinPrecessionODE
from .approximants.rom import OfficialSEOBNRv4ROM, SEOBNR_ROM
from .approximants.teobresums import TEOBResumS
from . import detectors
from .detectors import (
    Detector,
    DetectorNetwork,
    LISANetwork,
    PTANetwork,
    greenwich_mean_sidereal_time,
    get_psd,
    aLIGOZeroDetHighPower,
    aLIGOEarlyHighPower,
    AdvancedVirgo,
    KAGRA,
    EinsteinTelescope,
    CosmicExplorer,
    lisa_psd,
)
from . import inference
from .inference import (
    noise_weighted_inner_product,
    log_bessel_i0,
    optimal_snr,
    matched_filter_snr,
    WhittleLikelihood,
    fisher_information_matrix,
    covariance_matrix,
    sky_localization_area,
    optimize_map,
    RelativeBinningLikelihood,
    compute_relative_binning_bins,
    compute_relative_binning_summary,
)

__all__ = [
    "get_fd_waveform",
    "get_td_waveform",
    "WaveformGenerator",
    "MemoryWaveform",
    "compute_memory_h20",
    "memory_polarizations",
    "add_memory_to_waveform",
    "hybrid_binding_energy",
    "hybrid_meco_velocity",
    "hybrid_meco_frequency",
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
    "IMRPhenomXP",
    "IMRPhenomXPHM",
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
    "OfficialSEOBNRv4ROM",
    "SEOBNR_ROM",
    "TEOBResumS",
    "detectors",
    "Detector",
    "DetectorNetwork",
    "LISANetwork",
    "PTANetwork",
    "greenwich_mean_sidereal_time",
    "get_psd",
    "aLIGOZeroDetHighPower",
    "aLIGOEarlyHighPower",
    "AdvancedVirgo",
    "KAGRA",
    "EinsteinTelescope",
    "CosmicExplorer",
    "lisa_psd",
    "inference",
    "noise_weighted_inner_product",
    "log_bessel_i0",
    "optimal_snr",
    "matched_filter_snr",
    "WhittleLikelihood",
    "fisher_information_matrix",
    "covariance_matrix",
    "sky_localization_area",
    "optimize_map",
    "RelativeBinningLikelihood",
    "compute_relative_binning_bins",
    "compute_relative_binning_summary",
]

