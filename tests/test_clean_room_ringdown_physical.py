"""Tests for the source-defined physical equal-mirror ringdown subset."""

import pytest
import torch

from torchwave.approximants.kerr_ringdown import KerrRingdown, evaluate_qnm
from torchwave.approximants.kerr_ringdown_energy import (
    ringdown_amplitude_from_energy_fraction,
)
from torchwave.waveforms import get_fd_waveform


def test_energy_path_matches_explicit_native_pair_amplitude():
    mode = (2, 2, 0)
    mass, spin, distance, efficiency = 60.0, 0.7, 400.0, 0.01
    frequencies = torch.tensor([100.0, 200.0, 350.0], dtype=torch.float64)
    omega, tau, _ = evaluate_qnm(*mode, mass, spin)
    amplitude = ringdown_amplitude_from_energy_fraction(
        mass, distance, efficiency, omega, tau, convention="native_pair"
    )
    common = dict(
        final_mass=mass,
        final_spin=spin,
        phases={mode: 0.37},
        t0=0.002,
        inclination=0.9,
        phi=0.2,
        qnm_backend="independent",
        angular_basis="spheroidal",
    )
    explicit = KerrRingdown.frequency_domain(
        frequencies, amplitudes={mode: amplitude}, **common
    )
    physical = KerrRingdown.frequency_domain(
        frequencies,
        distance=distance,
        energy_fraction=efficiency,
        energy_mode=mode,
        **common,
    )
    for result, expected in zip(physical, explicit):
        torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


def test_energy_path_has_inverse_distance_scaling_and_is_wired_publicly():
    frequencies = torch.tensor([100.0, 200.0], dtype=torch.float64)
    kwargs = dict(
        approximant="ringdown",
        mass1=60.0,
        mass2=1.0,
        spin1z=0.7,
        distance=400.0,
        inclination=0.8,
        sample_frequencies=frequencies,
        energy_fraction=0.01,
        angular_basis="spheroidal",
    )
    near = get_fd_waveform(**kwargs)
    far = get_fd_waveform(**{**kwargs, "distance": 800.0})
    for near_pol, far_pol in zip(near, far):
        torch.testing.assert_close(near_pol, 2.0 * far_pol, rtol=2e-14, atol=1e-30)


def test_energy_path_rejects_underdetermined_conventions():
    frequencies = torch.tensor([100.0], dtype=torch.float64)
    common = dict(
        frequencies=frequencies,
        final_mass=60.0,
        final_spin=0.7,
        energy_fraction=0.01,
    )
    with pytest.raises(ValueError, match="distance is required"):
        KerrRingdown.frequency_domain(**common)
    with pytest.raises(ValueError, match="spheroidal"):
        KerrRingdown.frequency_domain(distance=400.0, **common)
    with pytest.raises(ValueError, match="mutually exclusive"):
        KerrRingdown.frequency_domain(
            distance=400.0,
            amplitudes={(2, 2, 0): 1.0},
            **common,
        )
    with pytest.raises(ValueError, match="energy_mode"):
        KerrRingdown.frequency_domain(
            distance=400.0,
            energy_mode=(2, 0, 0),
            angular_basis="spheroidal",
            **common,
        )
