"""torchwave: High-performance, clean-room gravitational-wave waveform generator in PyTorch."""

from torchwave.waveforms import get_fd_waveform
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.taylorf2_hm import TaylorF2_HM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenomp import IMRPhenomP

__version__ = "0.1.0"
__author__ = "Xan Morice-Atkinson"
__license__ = "MIT"

__all__ = [
    "get_fd_waveform",
    "TaylorF2",
    "TaylorF2_HM",
    "IMRPhenomD",
    "IMRPhenomP",
]
