"""Analytic check for the development-only Fehlberg output sampler."""

from dataclasses import replace

import torch

from dev.native_rom.eob.torch.adaptive_driver_clean import (
    _diagnostic_hermite_output, _diagnostic_rkf45_output,
)
from torchwave.approximants.eob.adaptive_driver_clean import integrate_adaptive


def _exponential_field(state):
    return state


def test_fehlberg_output_samples_preserve_accepted_trajectory_and_resolve_exponential():
    field = _exponential_field
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


def test_fehlberg_offgrid_local_error_has_fifth_order_step_accuracy():
    # A single accepted interval starts at the exact exponential state. Its
    # midpoint is off-grid with respect to the accepted trajectory, so this
    # isolates the fixed Fehlberg substep from accepted-state evolution.
    field = _exponential_field
    errors = []
    for interval in (0.4, 0.2, 0.1):
        run = integrate_adaptive(
            field, torch.ones(1, dtype=torch.float64), interval,
            t_end=interval, atol=1., rtol=1., initial_step=interval,
        )
        assert run.time.numel() == 2
        query = torch.tensor([interval/2], dtype=torch.float64)
        sample = _diagnostic_rkf45_output(replace(run, output_time=query), field)
        errors.append(float((sample.output_state[0, 0]-torch.exp(query[0])).abs()))

    assert errors[0] > errors[1] > errors[2] > 0
    assert errors[0]/errors[1] > 50
    assert errors[1]/errors[2] > 50
