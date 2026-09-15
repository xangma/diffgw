"""Run independent numerical checks for the Torch TD→FD conditioning layer.

The script uses analytic fixtures and, when installed, SciPy only as an
independent standard SOS filter oracle.  It never starts an EOB trajectory or
imports LALSuite.
"""

from __future__ import annotations

import math
import time

import torch

from torchwave.approximants.eob.td_fd_conditioning_clean import (
    _half_cosine_rise,
    apply_butterworth_highpass,
    condition_td,
    condition_td_to_fd,
    design_butterworth_highpass,
    fft_length_and_sampling,
    plan_td_support,
)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def main() -> None:
    print("Torch", torch.__version__, "dtype", torch.get_default_dtype())

    fixtures = (
        (30.0, 20.0, 20.0, 1.7156926398807189, 0.07484355785826285,
         16.900094009839023, 0.3215692639880719),
        (1.4, 1.4, 30.0, 56.080416766565115, 0.00419123924006272,
         28.430883710960526, 5.708041676656512),
    )
    print("Support fixtures (computed - contract):")
    for m1, m2, flow, chirp, plunge_rd, fstart, first_taper in fixtures:
        plan = plan_td_support(m1, m2, flow)
        computed = (_scalar(plan.chirp_time), _scalar(plan.plunge_ringdown_time),
                    _scalar(plan.start_frequency), _scalar(plan.first_taper_duration))
        target = (chirp, plunge_rd, fstart, first_taper)
        errors = tuple(a - b for a, b in zip(computed, target))
        print(f"  ({m1:g}+{m2:g}, {flow:g} Hz): {errors}")

    rise = _half_cosine_rise(torch.ones(8, dtype=torch.float64), torch.tensor(4))
    print("Half-cosine rise n=4:", rise.tolist())

    dt = 0.002
    f_low = 40.0
    sections = design_butterworth_highpass(dt, f_low)
    print("Filter corner (dimensionless):", _scalar(sections.corner))
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        scipy_signal = None
        print("SciPy SOS comparison: unavailable")
    if scipy_signal is not None:
        effective_corner_hz = math.atan(_scalar(sections.corner)) / (math.pi * dt)
        scipy_sos = scipy_signal.butter(
            8, effective_corner_hz, btype="highpass", fs=1.0 / dt, output="sos"
        )[::-1]
        generator = torch.Generator().manual_seed(20260914)
        x = torch.randn(4096, generator=generator, dtype=torch.float64)
        start = time.perf_counter()
        actual = apply_butterworth_highpass(x, dt, f_low)
        elapsed = time.perf_counter() - start
        expected = x.numpy()
        for row in scipy_sos:
            expected = scipy_signal.sosfilt(row[None, :], expected)
            expected = scipy_signal.sosfilt(row[None, :], expected[::-1])[::-1].copy()
        residual = actual.detach().numpy() - expected
        relative_l2 = float((sum(residual * residual) ** 0.5) /
                            (sum(expected * expected) ** 0.5))
        print("SciPy SOS comparison: rel_l2=%.3e max_abs=%.3e" %
              (relative_l2, float(abs(residual).max())))
        print("Torch recurrence timing: %.3f ms for N=4096" % (1000.0 * elapsed))
        long_signal = torch.randn(120000, generator=generator, dtype=torch.float64)
        start = time.perf_counter()
        long_output = apply_butterworth_highpass(long_signal, dt, f_low)
        long_elapsed = time.perf_counter() - start
        print("Torch recurrence timing: %.3f s for N=120000 finite=%s" %
              (long_elapsed, bool(torch.isfinite(long_output).all())))

    # Public synthetic TD fixture: enough samples for both taper stages and a
    # seven-sample exact paired suffix to exercise the post-filter trim.
    n = 520
    time_axis = torch.arange(n, dtype=torch.float32) * 1.0e-5
    plus = torch.sin(2.0 * torch.pi * 180.0 * time_axis)
    cross = 0.25 * torch.cos(2.0 * torch.pi * 130.0 * time_axis)
    plus = torch.cat((plus, torch.zeros(7)))
    cross = torch.cat((cross, torch.zeros(7)))
    plan = plan_td_support(1.4, 1.4, 800.0)
    conditioned = condition_td(
        plus, cross, dt=1.0e-5, support=plan, return_metadata=True
    )
    print("Conditioned fixture: input=%d output=%d trailing=%d first=%d second=(%d,%d)" % (
        plus.shape[-1], conditioned.h_plus.shape[-1],
        _scalar(conditioned.n_trailing_zeros), _scalar(conditioned.first_taper_samples),
        _scalar(conditioned.second_begin_samples), _scalar(conditioned.second_end_samples)))

    auto = fft_length_and_sampling(dt=1.0e-5, n_nonzero=conditioned.h_plus.shape[-1])
    explicit = fft_length_and_sampling(100.0, 1.0)
    print("FFT automatic: N=%d df=%.9g; explicit: N=%d df=%.9g" % (
        auto.n_time, _scalar(auto.delta_f), explicit.n_time, _scalar(explicit.delta_f)))

    tc = torch.tensor([3.0e-4], dtype=torch.float32, requires_grad=True)
    fd = condition_td_to_fd(
        plus.unsqueeze(0), cross.unsqueeze(0), dt=1.0e-5,
        support=plan, tc=tc, return_metadata=True,
    )
    loss = fd.h_plus.real.sum()
    loss.backward()
    print("End-to-end FD: shape=%s tc_grad=%+.6e finite=%s" % (
        tuple(fd.h_plus.shape), _scalar(tc.grad), bool(torch.isfinite(fd.h_plus).all())))


if __name__ == "__main__":
    main()
