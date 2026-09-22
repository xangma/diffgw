"""Waveform approximant implementations for torchwave."""

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.taylorf2_hm import TaylorF2_HM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomt import IMRPhenomT
from torchwave.approximants.imrphenomthm import IMRPhenomTHM
from torchwave.approximants.imrphenomtp import IMRPhenomTP
from torchwave.approximants.imrphenomtphm import IMRPhenomTPHM
from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.eob.waveform import SEOBNR_ODE
from torchwave.approximants.rom.surrogate import SEOBNR_ROM

from torchwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal,
    IMRPhenomD_NRTidalv2,
)
from torchwave.approximants.imrphenom_hm import (
    IMRPhenomHM,
)
from torchwave.approximants.imrphenom_xhm import (
    IMRPhenomXHM,
)
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS

from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.imrphenom_nsbh import IMRPhenomNSBH
from torchwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric
from torchwave.approximants.kerr_ringdown import (
    KerrRingdown, evaluate_qnm, get_ringdown_waveform,
    sampled_kerr_single_mode_reference, kerr_single_mode_reference,
    kerr_single_mode_fourier_reference,
)

from torchwave.approximants.precession_ode import SpinPrecessionODE

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
from torchwave.approximants.teobresums import (
    TEOBResumS,
    TEOBResumSHamiltonian,
    compute_tidal_merger_frequency,
)
from torchwave.approximants.continuous_wave import (
    ContinuousWave,
    continuous_wave_strain,
    continuous_wave_h0,
    continuous_wave_basis,
    continuous_wave_amplitudes,
    continuous_wave_phase,
)
from torchwave.approximants.memory import (
    compute_memory_h20,
    memory_polarizations,
    add_memory_to_waveform,
    MemoryWaveform,
)

__all__ = [
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
    "SpinPrecessionODE",
    "KerrRingdown",
    "evaluate_qnm",
    "get_ringdown_waveform",
    "sampled_kerr_single_mode_reference",
    "kerr_single_mode_reference",
    "kerr_single_mode_fourier_reference",
    "SEOBNR_ODE",
    "SEOBNR_ROM",
    "EMRIKludge",
    "kerr_geodesic_frequencies",
    "peters_mathews_radiation_reaction",
    "ParameterizedPostEinstein",
    "ppe_phase_shift",
    "modified_dispersion_phase_shift",
    "TEOBResumS",
    "TEOBResumSHamiltonian",
    "compute_tidal_merger_frequency",
    "ContinuousWave",
    "continuous_wave_strain",
    "continuous_wave_h0",
    "continuous_wave_basis",
    "continuous_wave_amplitudes",
    "continuous_wave_phase",
    "compute_memory_h20",
    "memory_polarizations",
    "add_memory_to_waveform",
    "MemoryWaveform",
]

