"""torchwave: High-performance, clean-room gravitational-wave waveform generator in PyTorch."""

from torchwave.waveforms import (
    get_fd_waveform,
    get_td_waveform,
    WaveformGenerator,
    list_approximants,
    normalize_waveform_params,
)
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.taylorf2_hm import TaylorF2_HM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomt import IMRPhenomT
from torchwave.approximants.imrphenomthm import IMRPhenomTHM
from torchwave.approximants.imrphenomtp import IMRPhenomTP
from torchwave.approximants.imrphenomtphm import IMRPhenomTPHM
from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal,
    IMRPhenomD_NRTidalv2,
)
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM
from torchwave.detectors.network import (
    Detector,
    DetectorNetwork,
    greenwich_mean_sidereal_time,
)

from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH
from torchwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric
from torchwave.approximants.kerr_ringdown import (
    KerrRingdown, get_ringdown_waveform,
    sampled_kerr_single_mode_reference, kerr_single_mode_reference,
    kerr_single_mode_fourier_reference,
)
from torchwave.detectors.psd import (
    aLIGOZeroDetHighPower,
    aLIGOEarlyHighPower,
    AdvancedVirgo,
    KAGRA,
    EinsteinTelescope,
    CosmicExplorer,
    get_psd,
)
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
from torchwave.approximants.precession_spintaylor import (
    SpinTaylorT4Engine,
    evaluate_spintaylor_precession,
)
from torchwave.approximants.precession_pnr import evaluate_pnr_precession
from torchwave.detectors.lisa import LISANetwork, lisa_psd
from torchwave.inference.fisher import (
    optimal_snr,
    matched_filter_snr,
    fisher_information_matrix,
    covariance_matrix,
    sky_localization_area,
    optimize_map,
)
from torchwave.inference.likelihood import (
    noise_weighted_inner_product,
    WhittleLikelihood,
)
from torchwave.inference.relative_binning import (
    RelativeBinningLikelihood,
    compute_relative_binning_bins,
    compute_relative_binning_summary,
)
from torchwave.inference.mcmc import (
    NUTS,
    gelman_rubin,
    leapfrog_step,
)
from torchwave.approximants.emri_kludge import (
    EMRIKludge,
    kerr_geodesic_frequencies,
    peters_mathews_radiation_reaction,
)
from torchwave.approximants.ppe import (
    ParameterizedPostEinstein,
    ppe_phase_shift,
    modified_dispersion_phase_shift,
)
from torchwave.detectors.pta import (
    PTANetwork,
    hellings_downs_correlation,
    stochastic_gwb_spectrum,
)
from torchwave import benchmark, external_data, profile, qnm

__version__ = "0.1.0"
__author__ = "Xan Morice-Atkinson"
__license__ = "MIT"

__all__ = [
    "get_fd_waveform",
    "get_td_waveform",
    "get_ringdown_waveform",
    "WaveformGenerator",
    "list_approximants",
    "normalize_waveform_params",
    "TaylorF2",
    "TaylorF2_HM",
    "TaylorF2_Eccentric",
    "IMRPhenomD",
    "IMRPhenomT",
    "IMRPhenomTHM",
    "IMRPhenomTP",
    "IMRPhenomTPHM",
    "IMRPhenomD_NRTidal",
    "IMRPhenomD_NRTidalv2",
    "IMRPhenomHM",
    "IMRPhenomXHM",
    "IMRPhenomXAS",
    "IMRPhenomP",
    "IMRPhenomXP",
    "IMRPhenomXPHM",
    "IMRPhenomNSBH",
    "SpinTaylorT4Engine",
    "evaluate_spintaylor_precession",
    "evaluate_pnr_precession",
    "KerrRingdown",
    "sampled_kerr_single_mode_reference",
    "kerr_single_mode_reference",
    "kerr_single_mode_fourier_reference",
    "Detector",
    "DetectorNetwork",
    "LISANetwork",
    "lisa_psd",
    "greenwich_mean_sidereal_time",
    "aLIGOZeroDetHighPower",
    "aLIGOEarlyHighPower",
    "AdvancedVirgo",
    "KAGRA",
    "EinsteinTelescope",
    "CosmicExplorer",
    "get_psd",
    "noise_weighted_inner_product",
    "WhittleLikelihood",
    "optimal_snr",
    "matched_filter_snr",
    "fisher_information_matrix",
    "covariance_matrix",
    "sky_localization_area",
    "optimize_map",
    "RelativeBinningLikelihood",
    "compute_relative_binning_bins",
    "compute_relative_binning_summary",
    "NUTS",
    "gelman_rubin",
    "leapfrog_step",
    "EMRIKludge",
    "kerr_geodesic_frequencies",
    "peters_mathews_radiation_reaction",
    "ParameterizedPostEinstein",
    "ppe_phase_shift",
    "modified_dispersion_phase_shift",
    "PTANetwork",
    "hellings_downs_correlation",
    "stochastic_gwb_spectrum",
    "benchmark",
    "external_data",
    "profile",
    "qnm",
]
