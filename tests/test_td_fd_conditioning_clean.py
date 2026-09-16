"""Independent checks for the Torch-only TD→FD conditioning layer."""

import math

import pytest
import torch

from torchwave.approximants.eob.td_fd_conditioning_clean import (
    _half_cosine_rise,
    _second_stage_taper,
    apply_butterworth_highpass,
    apply_tc_phase,
    condition_td,
    condition_td_to_fd,
    count_trailing_zero_padding,
    design_butterworth_highpass,
    fft_length_and_sampling,
    pair_leading_pad_final_crop,
    plan_td_support,
)


def test_support_planner_matches_public_contract_fixtures():
    expected = (
        (30.0, 20.0, 20.0, 1.7156926398807189, 0.07484355785826285,
         16.900094009839023, 0.3215692639880719),
        (1.4, 1.4, 30.0, 56.080416766565115, 0.00419123924006272,
         28.430883710960526, 5.708041676656512),
    )
    for m1, m2, flow, chirp, plunge_rd, fstart, taper in expected:
        plan = plan_td_support(m1, m2, flow)
        def as_tensor(value):
            return torch.tensor(value, dtype=plan.chirp_time.dtype)

        torch.testing.assert_close(plan.chirp_time, as_tensor(chirp), rtol=2e-13, atol=2e-15)
        torch.testing.assert_close(plan.plunge_ringdown_time, as_tensor(plunge_rd), rtol=2e-13, atol=2e-15)
        torch.testing.assert_close(plan.start_frequency, as_tensor(fstart), rtol=2e-13, atol=2e-15)
        torch.testing.assert_close(plan.first_taper_duration, as_tensor(taper), rtol=2e-13, atol=2e-15)


def test_tapers_use_half_cosine_and_contract_end_convention():
    signal = torch.ones(12, dtype=torch.float64)
    rise = _half_cosine_rise(signal, torch.tensor(4))
    expected_rise = torch.tensor(
        [0.0, (2.0 - math.sqrt(2.0)) / 4.0, 0.5,
         (2.0 + math.sqrt(2.0)) / 4.0] + [1.0] * 8,
        dtype=torch.float64,
    )
    torch.testing.assert_close(rise, expected_rise, rtol=0.0, atol=2e-15)

    # One cycle at each end is four samples here.  The last retained sample
    # is one step before the zero that would follow the data.
    tapered, begin, end, applied = _second_stage_taper(
        signal,
        torch.tensor(100.0, dtype=torch.float64),
        torch.tensor(100.0, dtype=torch.float64),
        torch.tensor(0.01, dtype=torch.float64),
    )
    assert applied
    assert begin.item() == 4 and end.item() == 4
    torch.testing.assert_close(tapered[:4], expected_rise[:4], rtol=0.0, atol=2e-15)
    expected_end = torch.tensor(
        [1.0, expected_rise[3], expected_rise[2], expected_rise[1]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(tapered[-4:], expected_end, rtol=0.0, atol=2e-15)


def test_butterworth_matches_scipy_sosfilt_with_independent_standard_sections():
    scipy_signal = pytest.importorskip("scipy.signal")
    dt = 0.002
    f_low = 40.0
    sections = design_butterworth_highpass(dt, f_low)
    # Convert the independently designed normalized analog corner back to a
    # physical digital corner for scipy's standard Butterworth constructor.
    effective_corner_hz = math.atan(float(sections.corner)) / (math.pi * dt)
    scipy_sos = scipy_signal.butter(
        8, effective_corner_hz, btype="highpass", fs=1.0 / dt, output="sos"
    )[::-1]
    x = torch.linspace(-0.7, 1.1, 257, dtype=torch.float64)
    actual = apply_butterworth_highpass(x, dt, f_low)

    expected = x.numpy()
    for row in scipy_sos:
        expected = scipy_signal.sosfilt(row[None, :], expected)
        expected = scipy_signal.sosfilt(row[None, :], expected[::-1])[::-1].copy()
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=3e-12, atol=3e-13)


def test_conditioning_trims_paired_padding_batches_and_has_finite_gradients():
    dtype = torch.float32
    batch, n = 2, 520
    time = torch.arange(n, dtype=dtype) * 1.0e-5
    h_plus = torch.stack((torch.sin(2.0 * torch.pi * 180.0 * time),
                          torch.cos(2.0 * torch.pi * 240.0 * time))).requires_grad_()
    h_cross = 0.25 * torch.stack((torch.cos(2.0 * torch.pi * 130.0 * time),
                                  torch.sin(2.0 * torch.pi * 210.0 * time)))
    h_plus_pad = torch.cat((h_plus, torch.zeros(batch, 7, dtype=dtype)), dim=-1)
    h_cross_pad = torch.cat((h_cross, torch.zeros(batch, 7, dtype=dtype)), dim=-1)
    plan = plan_td_support(
        torch.tensor([1.4, 1.5], dtype=torch.float64),
        torch.tensor([1.4, 1.5], dtype=torch.float64),
        torch.tensor([800.0, 800.0], dtype=torch.float64),
    )
    conditioned = condition_td(
        h_plus_pad,
        h_cross_pad,
        dt=torch.tensor(1.0e-5, dtype=dtype),
        support=plan,
        return_metadata=True,
    )
    assert conditioned.h_plus.shape == (batch, n)
    assert conditioned.h_cross.shape == (batch, n)
    assert torch.equal(conditioned.n_trailing_zeros, torch.tensor([7, 7]))
    assert conditioned.second_stage_applied
    assert torch.isfinite(conditioned.h_plus).all()
    assert torch.isfinite(conditioned.h_cross).all()
    loss = conditioned.h_plus.square().mean() + conditioned.h_cross.square().mean()
    loss.backward()
    assert h_plus.grad is not None and torch.isfinite(h_plus.grad).all()


def test_fft_rules_pairing_and_tc_phase():
    explicit = fft_length_and_sampling(100.0, 1.0)
    assert explicit.n_time == 256
    assert explicit.n_frequency == 129
    assert explicit.f_nyquist.item() == 128.0
    torch.testing.assert_close(explicit.dt, torch.tensor(1.0 / 256.0, dtype=explicit.dt.dtype))

    automatic = fft_length_and_sampling(dt=0.1, n_nonzero=8)
    assert automatic.n_time == 16  # strictly greater than an existing power of two

    plus = torch.tensor([1.0, 2.0, 3.0])
    cross = -plus
    padded, padded_cross = pair_leading_pad_final_crop(plus, cross, 5)
    torch.testing.assert_close(padded, torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(padded_cross, -padded)
    cropped, _ = pair_leading_pad_final_crop(padded, padded_cross, 3)
    torch.testing.assert_close(cropped, plus)

    fd = torch.tensor([1.0 + 0.5j, -0.25 + 0.75j], dtype=torch.complex128)
    freq = torch.tensor([0.0, 10.0], dtype=torch.float64)
    tc = torch.tensor(0.125, dtype=torch.float64, requires_grad=True)
    shifted, _ = apply_tc_phase(fd, fd, freq, tc)
    torch.testing.assert_close(shifted[0], fd[0])
    torch.testing.assert_close(shifted[1], fd[1] * torch.exp(torch.tensor(-2j * math.pi * 10.0 * 0.125)))
    shifted.real.sum().backward()
    assert tc.grad is not None and torch.isfinite(tc.grad)


def test_end_to_end_auto_fft_keeps_batch_and_separate_tc_parameter():
    n = 520
    time = torch.arange(n, dtype=torch.float32) * 1.0e-5
    plus = torch.sin(2.0 * torch.pi * 170.0 * time).unsqueeze(0)
    cross = torch.cos(2.0 * torch.pi * 230.0 * time).unsqueeze(0)
    tc = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    result = condition_td_to_fd(
        plus,
        cross,
        dt=1.0e-5,
        mass1=1.4,
        mass2=1.4,
        f_low=800.0,
        tc=tc,
        return_metadata=True,
    )
    assert result.sampling.n_time == 1024
    assert result.h_plus.shape == (1, 513)
    assert torch.equal(result.frequencies, result.sampling.frequencies())
    loss = result.h_plus.real.sum()
    loss.backward()
    assert tc.grad is not None and torch.isfinite(tc.grad).all()


def test_trailing_zero_count_requires_exact_paired_suffix():
    plus = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    cross = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    counts = count_trailing_zero_padding(plus, cross)
    assert torch.equal(counts, torch.tensor([2, 0]))
