"""Reduced-Order Model (ROM / Surrogate) waveform engines in JAX."""

from jaxwave.approximants.rom.surrogate import SEOBNR_ROM
from jaxwave.approximants.rom.official import (
    OFFICIAL_ROM_FILENAME,
    OFFICIAL_ROM_SHA256,
    ROMFileSchema,
    ROMSubmodelSchema,
    inspect_official_rom,
    resolve_official_rom_path,
    sha256_dataset,
    sha256_file,
)
from jaxwave.approximants.rom.bspline import (
    OfficialROMSurrogate,
    bspline_basis,
    load_official_rom,
    cubic_spline_interpolate,
    evaluate_stitched_frequency,
    linear_interpolate,
    reconstruct_reduced_basis,
    tensor_product_basis,
)
from jaxwave.approximants.rom.official_wrapper import (
    OfficialSEOBNRv4ROM,
    independent_ringdown_mf,
    table_convention_ringdown_mf,
)

__all__ = [
    "SEOBNR_ROM",
    "OFFICIAL_ROM_FILENAME",
    "OFFICIAL_ROM_SHA256",
    "ROMFileSchema",
    "ROMSubmodelSchema",
    "inspect_official_rom",
    "resolve_official_rom_path",
    "sha256_dataset",
    "sha256_file",
    "bspline_basis",
    "tensor_product_basis",
    "reconstruct_reduced_basis",
    "OfficialROMSurrogate",
    "load_official_rom",
    "linear_interpolate",
    "cubic_spline_interpolate",
    "evaluate_stitched_frequency",
    "OfficialSEOBNRv4ROM",
    "independent_ringdown_mf",
    "table_convention_ringdown_mf",
]
