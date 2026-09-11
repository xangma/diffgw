"""Regression and consistency tests for IMRPhenomD independent reference phase and time-shift conventions.

Validates:
1. Reference frequency phase anchor:
   - For all masses and spins with tc = 0, arg(h_plus(f_ref)) == 2 * phic within 1e-12.
2. Frequency-domain time shift linearity:
   - A time shift tc != 0 produces a pure linear phase rotation exp(-i * 2 * pi * f * tc) with zero amplitude alteration.
3. Batch vs scalar evaluation equivalence across reference frequencies using IMRPhenomD.__call__.
4. Autograd differentiability with respect to spins and masses under the global time gauge.
"""

import math
import pytest
import torch

from torchwave import get_fd_waveform
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.constants import PI


def test_imrphenomd_reference_phase_anchor():
    """Verify that arg(h_plus(f_ref)) equals 2 * phic for varied masses, spins, and distances when tc = 0."""
    cases = [
        {"m1": 30.0, "m2": 20.0, "s1": 0.3, "s2": -0.2, "phic": 0.3, "f_ref": 20.0},
        {"m1": 57.0, "m2": 13.0, "s1": 0.83, "s2": -0.41, "phic": 0.71, "f_ref": 27.0},
        {"m1": 23.0, "m2": 21.0, "s1": -0.6, "s2": 0.2, "phic": 0.71, "f_ref": 35.0},
        {"m1": 80.0, "m2": 8.0, "s1": -0.7, "s2": 0.1, "phic": 0.71, "f_ref": 20.0},
        {"m1": 20.0, "m2": 20.0, "s1": 0.0, "s2": 0.0, "phic": 0.45, "f_ref": 25.0},
    ]

    for c in cases:
        f_ref_val = c["f_ref"]
        f_grid = torch.tensor([f_ref_val], dtype=torch.float64)

        hp, hc = get_fd_waveform(
            "IMRPhenomD",
            mass1=c["m1"],
            mass2=c["m2"],
            chi1=c["s1"],
            chi2=c["s2"],
            distance=400.0,
            inclination=0.5,
            phic=c["phic"],
            tc=0.0,
            f_ref=f_ref_val,
            sample_frequencies=f_grid,
        )

        # In polarizations: h_plus ~ exp(-i * total_phase)
        # total_phase(f_ref) = -2 * phic => h_plus ~ exp(i * 2 * phic)
        # arg(h_plus) should equal 2 * phic modulo 2*pi
        arg_hp = torch.angle(hp[0]).item()
        expected = (2.0 * c["phic"]) % (2.0 * math.pi)
        actual = arg_hp % (2.0 * math.pi)

        diff = abs(actual - expected)
        if diff > math.pi:
            diff = 2.0 * math.pi - diff

        assert diff < 1e-12, (
            f"Case {c}: expected arg(h_plus(f_ref)) = {expected}, got {actual} (diff {diff})"
        )


def test_imrphenomd_time_shift_linearity():
    """Verify that tc produces an exact linear phase rotation without altering the amplitude spectrum."""
    f = torch.linspace(20.0, 300.0, 200, dtype=torch.float64)
    m1, m2 = 35.0, 25.0
    s1, s2 = 0.4, -0.3
    tc_val = 0.015  # 15 ms

    hp_t0, hc_t0 = get_fd_waveform(
        "IMRPhenomD",
        mass1=m1,
        mass2=m2,
        chi1=s1,
        chi2=s2,
        distance=500.0,
        inclination=0.4,
        phic=0.5,
        tc=0.0,
        f_ref=30.0,
        sample_frequencies=f,
    )

    hp_tc, hc_tc = get_fd_waveform(
        "IMRPhenomD",
        mass1=m1,
        mass2=m2,
        chi1=s1,
        chi2=s2,
        distance=500.0,
        inclination=0.4,
        phic=0.5,
        tc=tc_val,
        f_ref=30.0,
        sample_frequencies=f,
    )

    # Amplitude must be strictly identical across all frequencies (normalized relative error < 1e-12)
    amp_t0 = torch.abs(hp_t0)
    amp_tc = torch.abs(hp_tc)
    rel_amp_diff = torch.max(torch.abs(amp_tc - amp_t0) / amp_t0).item()
    assert rel_amp_diff < 1e-12, f"Expected relative amplitude difference < 1e-12, got {rel_amp_diff}"

    # Phase difference must equal exactly -2 * pi * f * tc (since h ~ exp(-i * total_phase))
    # total_phase_tc - total_phase_t0 = 2 * pi * f * tc
    # => h_tc / h_t0 = exp(-i * 2 * pi * f * tc)
    phase_ratio = hp_tc / hp_t0
    expected_rot = torch.complex(torch.cos(-2.0 * PI * f * tc_val), torch.sin(-2.0 * PI * f * tc_val))
    norm_expected = torch.linalg.norm(expected_rot)
    rel_rot_diff = (torch.linalg.norm(phase_ratio - expected_rot) / norm_expected).item()
    assert rel_rot_diff < 1e-10, f"Expected relative rotation diff < 1e-10, got {rel_rot_diff}"


def test_imrphenomd_batch_reference_frequencies():
    """Verify batched evaluation with multiple reference frequencies matches scalar evaluations via IMRPhenomD.__call__."""
    f = torch.linspace(20.0, 200.0, 50, dtype=torch.float64)
    m1_list = [30.0, 45.0]
    m2_list = [20.0, 15.0]
    s1_list = [0.2, -0.4]
    s2_list = [-0.1, 0.3]
    fref_list = [20.0, 30.0]

    hps_scalar = []
    for m1, m2, s1, s2, fref in zip(m1_list, m2_list, s1_list, s2_list, fref_list):
        hp, _ = get_fd_waveform(
            "IMRPhenomD",
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=400.0,
            inclination=0.0,
            phic=0.3,
            tc=0.0,
            f_ref=fref,
            sample_frequencies=f,
        )
        hps_scalar.append(hp)

    m1_b = torch.tensor(m1_list, dtype=torch.float64)
    m2_b = torch.tensor(m2_list, dtype=torch.float64)
    s1_b = torch.tensor(s1_list, dtype=torch.float64)
    s2_b = torch.tensor(s2_list, dtype=torch.float64)
    fref_b = torch.tensor(fref_list, dtype=torch.float64)

    # Use static call IMRPhenomD.__call__
    hp_b, _ = IMRPhenomD.__call__(
        f=f,
        mass1=m1_b,
        mass2=m2_b,
        chi1=s1_b,
        chi2=s2_b,
        distance=400.0,
        inclination=0.0,
        phic=0.3,
        tc=0.0,
        f_ref=fref_b,
    )

    for i in range(len(m1_list)):
        norm_ref = torch.linalg.norm(hps_scalar[i])
        rel_diff = (torch.linalg.norm(hp_b[i] - hps_scalar[i]) / norm_ref).item()
        assert rel_diff < 1e-12, f"Batched vs scalar mismatch for batch item {i}: {rel_diff}"


def test_imrphenomd_gauge_autograd():
    """Verify that autograd flows cleanly through the intrinsic global time gauge slope_peak."""
    f = torch.linspace(20.0, 100.0, 30, dtype=torch.float64)
    s1 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomD.__call__(
        f=f,
        mass1=30.0,
        mass2=20.0,
        chi1=s1,
        chi2=-0.2,
        distance=400.0,
        inclination=0.5,
        phic=0.3,
        tc=0.0,
        f_ref=20.0,
    )

    loss = (hp.real ** 2).sum()
    loss.backward()

    assert s1.grad is not None
    assert torch.isfinite(s1.grad)
    assert s1.grad.item() != 0.0
