"""Effective-One-Body (EOB) approximants and dynamical components in JAX."""

from jaxwave.approximants.eob.hamiltonian import EOBHamiltonian
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian, DeformedKerrPotentials
from jaxwave.approximants.eob.radiation_reaction import EOBRadiationReaction, FactorizedRadiationReaction
from jaxwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from jaxwave.approximants.eob.events import OrbitalFrequencyPeak, orbital_frequency_peak
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator, FrequencyPeakTrajectory
from jaxwave.approximants.eob.merger_ringdown import (
    MergerRingdownShape,
    seobnrv4_merger_shape,
    merger_ringdown_22,
)
from jaxwave.approximants.eob.continuous_matching import continuous_spline_peak
from jaxwave.approximants.eob.numerical_attachment import (
    NaturalCubicSpline,
    natural_cubic_spline,
)
from jaxwave.approximants.eob.nqc import (
    NQCCoefficients,
    NQCMatchingTargets,
    NQCCalibration,
    seobnrv4_nqc_parameters,
    seobnrv4_nqc_targets,
    seobnrv4_nqc_time_offset,
    seobnrv4_nqc_time_offset_numerical_compatibility,
    nqc_22_basis,
    nqc_22_factor,
    calibrate_nqc_22,
)
from jaxwave.approximants.eob.attachment import AttachedMode22, attach_seobnrv4_22
from jaxwave.approximants.eob.waveform import SEOBNR_ODE, _interp1d, _quadrupole_fd
from jaxwave.approximants.eob.published_waveform import (
    PublishedEOB22,
    published_eob22_to_fd,
    reconstruct_seobnrv4_22,
    SEOBNRv4Published,
    SEOBNRv4PublishedNonspinning,
)
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin

__all__ = [
    "EOBHamiltonian",
    "SEOBNRv4Hamiltonian",
    "DeformedKerrPotentials",
    "EOBRadiationReaction",
    "FactorizedRadiationReaction",
    "find_quasicircular_initial_conditions",
    "EOBTrajectoryIntegrator",
    "FrequencyPeakTrajectory",
    "OrbitalFrequencyPeak",
    "orbital_frequency_peak",
    "MergerRingdownShape",
    "seobnrv4_merger_shape",
    "merger_ringdown_22",
    "continuous_spline_peak",
    "NaturalCubicSpline",
    "natural_cubic_spline",
    "NQCCoefficients",
    "NQCMatchingTargets",
    "NQCCalibration",
    "seobnrv4_nqc_parameters",
    "seobnrv4_nqc_targets",
    "seobnrv4_nqc_time_offset",
    "seobnrv4_nqc_time_offset_numerical_compatibility",
    "nqc_22_basis",
    "nqc_22_factor",
    "calibrate_nqc_22",
    "AttachedMode22",
    "attach_seobnrv4_22",
    "SEOBNR_ODE",
    "_interp1d",
    "_quadrupole_fd",
    "PublishedEOB22",
    "published_eob22_to_fd",
    "reconstruct_seobnrv4_22",
    "SEOBNRv4Published",
    "SEOBNRv4PublishedNonspinning",
    "eob_final_mass_fraction",
    "hofmann_final_spin",
]
