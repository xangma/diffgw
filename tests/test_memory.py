"""Tests for Non-Linear Gravitational Wave Memory (Christodoulou Memory).

Clean-room literature benchmarks:
- Christodoulou (1991), Phys. Rev. Lett. 67, 1486
- Favata (2010), Class. Quantum Grav. 27, 084036 (arXiv:1003.3486)
- Talbot et al. (2018), Phys. Rev. D 98, 084031 (arXiv:1807.00990)
"""

import math
import pytest
import torch

from torchwave.approximants.memory import (
    compute_memory_h20,
    memory_polarizations,
    add_memory_to_waveform,
    MemoryWaveform,
)


def test_memory_monotonic_and_positive_saturation():
    """Verify that h_20^mem(t) is monotonically increasing (d h_20^mem / dt >= 0) and saturates to positive constant."""
    # Construct a realistic damped chirping signal representing inspiral-merger-ringdown
    t = torch.linspace(-0.1, 0.1, 1000, dtype=torch.float64)
    dt = float((t[1] - t[0]).item())

    # Chirp amplitude envelope peaking at t = 0 and rapidly decaying (merger-ringdown)
    amp = 1e-21 * torch.exp(-((t / 0.01) ** 2))
    phase = 2.0 * math.pi * 150.0 * t + 500.0 * (t ** 2)

    hp = amp * torch.cos(phase)
    hc = amp * torch.sin(phase)

    h20_mem = compute_memory_h20(times=t, modes_or_hp=hp, hc=hc)

    # 1. Monotonicity check: differences between consecutive points must be non-negative (within floating point precision)
    diffs = torch.diff(h20_mem)
    assert torch.all(diffs >= -1e-32), "Memory must be monotonically non-decreasing"

    # 2. Positive constant saturation post-merger:
    # At late times (t > 0.05), the oscillatory signal has decayed to 0, so h20_mem must remain constant and positive
    late_times_idx = t > 0.05
    late_memory = h20_mem[late_times_idx]

    saturation_val = late_memory[-1].item()
    assert saturation_val > 0.0, "Net memory offset Delta h^mem must be strictly positive"

    # Check flatness at late times
    assert torch.allclose(late_memory, late_memory[-1:], atol=1e-28), "Memory must saturate to constant offset post-merger"


def test_memory_polarization_angular_dependence():
    """Verify that memory polarizations satisfy h_+^mem proportional to sin^2(iota) and h_x^mem == 0."""
    t = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
    h20 = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)

    # Face-on (iota = 0): sin^2(0) = 0 -> hp_mem = 0
    hp_face_on, hc_face_on = memory_polarizations(h20, inclination=0.0)
    assert torch.allclose(hp_face_on, torch.zeros_like(hp_face_on))
    assert torch.allclose(hc_face_on, torch.zeros_like(hc_face_on))

    # Edge-on (iota = pi/2): sin^2(pi/2) = 1 -> maximal memory
    hp_edge_on, hc_edge_on = memory_polarizations(h20, inclination=math.pi / 2.0)
    assert torch.all(hp_edge_on >= 0.0)
    assert hp_edge_on[-1] > 0.0
    assert torch.allclose(hc_edge_on, torch.zeros_like(hc_edge_on))

    # General inclination: hp_mem(iota) = hp_edge_on * sin^2(iota)
    iota_val = 0.6
    hp_inc, hc_inc = memory_polarizations(h20, inclination=iota_val)
    expected_hp = hp_edge_on * (math.sin(iota_val) ** 2)
    assert torch.allclose(hp_inc, expected_hp, atol=1e-12)
    assert torch.allclose(hc_inc, torch.zeros_like(hc_inc))


def test_add_memory_to_waveform_api():
    """Verify add_memory_to_waveform utility and module wrapper."""
    t = torch.linspace(-0.05, 0.05, 500, dtype=torch.float64)
    hp = 1e-21 * torch.cos(2.0 * math.pi * 100.0 * t) * torch.exp(-((t / 0.01) ** 2))
    hc = 1e-21 * torch.sin(2.0 * math.pi * 100.0 * t) * torch.exp(-((t / 0.01) ** 2))

    hp_tot, hc_tot, h20 = add_memory_to_waveform(
        times=t,
        hp=hp,
        hc=hc,
        inclination=math.pi / 3.0,
    )

    assert hp_tot.shape == hp.shape
    assert hc_tot.shape == hc.shape
    assert h20.shape == t.shape
    assert not torch.allclose(hp_tot, hp, atol=1e-30), "Memory must perturb h_plus"
    assert torch.allclose(hc_tot, hc, atol=1e-30), "h_cross should remain unaffected by memory in planar orbits"

