"""Reduced-Order Model (ROM / Surrogate) waveform engines."""

from torchwave.approximants.rom.surrogate import SEOBNR_ROM, HDF5ROMSurrogate
from torchwave.approximants.rom.generator import (
    generate_eob_training_grid,
    compute_svd_reduced_basis,
    fit_parameter_mapping,
    train_and_export_surrogate,
)

__all__ = [
    "SEOBNR_ROM",
    "HDF5ROMSurrogate",
    "generate_eob_training_grid",
    "compute_svd_reduced_basis",
    "fit_parameter_mapping",
    "train_and_export_surrogate",
]

