"""Effective-One-Body (EOB) approximants and dynamical components."""

from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.waveform import SEOBNR_ODE
from torchwave.approximants.eob.published_waveform import (
    SEOBNRv4Published, SEOBNRv4PublishedNonspinning, reconstruct_seobnrv4_22)
from torchwave.approximants.eob.adaptive_driver_clean import (
    AdaptiveIntegrationError, AdaptiveStatistics, AdaptiveTrajectory,
    AlignedSpinV4Event, SplicedNumericalAttachment, TwoStageEOBTrajectory,
    integrate_adaptive, resample_natural_cubic,
    reconstruct_seobnrv4_22_adaptive,
    reconstruct_seobnrv4_22_numerical_compatibility)
from torchwave.approximants.eob.numerical_attachment import (
    NaturalCubicSpline, FullSeriesNQC, NumericalAttachedMode22,
    natural_cubic_spline, calibrate_nqc_22_full_series,
    seobnrv4_internal_step, attach_seobnrv4_22_numerical_compatibility)
from torchwave.approximants.eob.teob_postmerger import (
    merger_amplitude as teob_merger_amplitude,
    merger_frequency as teob_merger_frequency,
    nonspinning_ringdown_shape as teob_nonspinning_ringdown_shape,
    nqc_targets as teob_nqc_targets,
    postmerger_22 as teob_postmerger_22,
    qnm_fits as teob_qnm_fits,
)
from torchwave.approximants.eob.teob_uib2016_remnant import (
    uib2016_nonspinning_final_mass_fraction,
    uib2016_nonspinning_final_spin,
    uib2016_nonspinning_radiated_energy,
    uib2016_nonspinning_remnant,
)
from torchwave.approximants.eob.teob_uib2016_postmerger import (
    uib2016_nonspinning_postmerger_22,
)
from torchwave.approximants.eob.teob_trajectory import (
    TEOBResumSTrajectoryQuantities,
    conservative_trajectory_quantities,
)
from torchwave.approximants.eob.teob_published_driver import (
    TEOB22Result, teob22_mode, teob22_flux,
    teob22_mode_and_frequency, integrate_teob22,
)

__all__ = [
    "EOBHamiltonian",
    "EOBRadiationReaction",
    "find_quasicircular_initial_conditions",
    "EOBTrajectoryIntegrator",
    "SEOBNR_ODE",
    "SEOBNRv4Published",
    "SEOBNRv4PublishedNonspinning",
    "reconstruct_seobnrv4_22",
    "AdaptiveIntegrationError",
    "AdaptiveStatistics",
    "AdaptiveTrajectory",
    "AlignedSpinV4Event",
    "SplicedNumericalAttachment",
    "TwoStageEOBTrajectory",
    "integrate_adaptive",
    "resample_natural_cubic",
    "reconstruct_seobnrv4_22_adaptive",
    "reconstruct_seobnrv4_22_numerical_compatibility",
    "NaturalCubicSpline",
    "FullSeriesNQC",
    "NumericalAttachedMode22",
    "natural_cubic_spline",
    "calibrate_nqc_22_full_series",
    "seobnrv4_internal_step",
    "attach_seobnrv4_22_numerical_compatibility",
    "teob_merger_amplitude",
    "teob_merger_frequency",
    "teob_nonspinning_ringdown_shape",
    "teob_nqc_targets",
    "teob_postmerger_22",
    "teob_qnm_fits",
    "uib2016_nonspinning_radiated_energy",
    "uib2016_nonspinning_final_mass_fraction",
    "uib2016_nonspinning_final_spin",
    "uib2016_nonspinning_remnant",
    "uib2016_nonspinning_postmerger_22",
    "TEOBResumSTrajectoryQuantities",
    "conservative_trajectory_quantities",
    "TEOB22Result",
    "teob22_mode",
    "teob22_flux",
    "teob22_mode_and_frequency",
    "integrate_teob22",
]
