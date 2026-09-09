"""Waveform approximant implementations for torchwave."""

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.eob.waveform import SEOBNR_ODE
from torchwave.approximants.rom.surrogate import SEOBNR_ROM

__all__ = [
    "TaylorF2",
    "IMRPhenomD",
    "SEOBNR_ODE",
    "SEOBNR_ROM",
]
