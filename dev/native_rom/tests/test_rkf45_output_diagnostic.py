"""Analytic check for the development-only Fehlberg output sampler."""

import torch

from dev.native_rom.eob.torch.adaptive_driver_clean import (
    _diagnostic_hermite_output, _diagnostic_rkf45_output,
)
from torchwave.approximants.eob.adaptive_driver_clean import integrate_adaptive


def test_fehlberg_output_samples_preserve_accepted_trajectory_and_resolve_exponential():
    field = lambda state: state
    run = integrate_adaptive(
        field, torch.ones(1, dtype=torch.float64), 0.2,
        t_end=1.0, atol=1e-9, rtol=1e-8,
    )
    sampled = _diagnostic_rkf45_output(run, field)
    hermite = _diagnostic_hermite_output(run)
    exact = torch.exp(run.output_time)
    rkf_error = (sampled.output_state[:, 0] - exact).abs().max()
    hermite_error = (hermite.output_state[:, 0] - exact).abs().max()

    assert rkf_error < 2e-8
    assert rkf_error < hermite_error
    assert sampled.time is run.time
    assert sampled.state is run.state
    assert sampled.derivative is run.derivative
    assert sampled.statistics is run.statistics
