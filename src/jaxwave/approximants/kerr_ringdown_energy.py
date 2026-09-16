"""Exact energy normalization for an isolated damped Kerr mode or mirror pair in JAX.

Port of torchwave.approximants.kerr_ringdown_energy for JAX / jaxwave.
"""

from jaxwave.approximants.kerr_ringdown import ringdown_amplitude_from_energy_fraction

__all__ = ["ringdown_amplitude_from_energy_fraction"]
