import inspect
from dataclasses import dataclass

import torch
import pytest

from dev.native_rom.eob.torch.adaptive_driver_clean import (
    _diagnostic_hermite_output, reconstruct_seobnrv4_22_adaptive,
)


DTYPE = torch.float64


def test_adaptive_compatibility_defaults_are_explicitly_reviewed_choices():
    signature = inspect.signature(reconstruct_seobnrv4_22_adaptive)
    assert signature.parameters["spin_truncation"].default == "generic-leading-ell8"
    assert signature.parameters["final_spin_fit_order"].default == (3, 4)
    omega = torch.tensor(0.05, dtype=DTYPE)
    with pytest.raises(ValueError, match="spin_truncation"):
        reconstruct_seobnrv4_22_adaptive(omega, 30.0, 30.0,
                                          spin_truncation="unknown")
    with pytest.raises(ValueError, match="diagnostic_mode_43_higher_spin"):
        reconstruct_seobnrv4_22_adaptive(omega, 30.0, 30.0,
                                          diagnostic_mode_43_higher_spin="unknown")
    with pytest.raises(ValueError, match="final_spin_fit_order"):
        reconstruct_seobnrv4_22_adaptive(omega, 30.0, 30.0,
                                          final_spin_fit_order=(1, 2))


def test_diagnostic_grid_overrides_reject_invalid_values_before_solve():
    omega = torch.tensor(0.05, dtype=DTYPE)
    with pytest.raises(ValueError, match="diagnostic_coarse_initial_step_M"):
        reconstruct_seobnrv4_22_adaptive(
            omega, 30.0, 30.0, diagnostic_coarse_initial_step_M=0.0)
    with pytest.raises(ValueError, match="diagnostic_restart_time_M must lie"):
        reconstruct_seobnrv4_22_adaptive(
            omega, 30.0, 30.0, dt_geom=0.25,
            diagnostic_restart_time_M=165.125)
    with pytest.raises(TypeError, match="diagnostic_hermite_resampling"):
        reconstruct_seobnrv4_22_adaptive(
            omega, 30.0, 30.0, diagnostic_hermite_resampling="yes")
    with pytest.raises(ValueError, match="diagnostic_fine_max_step_M"):
        reconstruct_seobnrv4_22_adaptive(
            omega, 30.0, 30.0, diagnostic_fine_max_step_M=0.0)


def test_diagnostic_hermite_resampling_reproduces_cubic_on_nonuniform_knots():
    @dataclass(frozen=True)
    class Run:
        time: torch.Tensor
        state: torch.Tensor
        derivative: torch.Tensor
        output_time: torch.Tensor
        output_state: torch.Tensor

    knots = torch.tensor([0.0, 0.2, 1.0], dtype=DTYPE)
    query = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=DTYPE)
    run = Run(knots, (knots**3).unsqueeze(-1), (3*knots**2).unsqueeze(-1),
              query, torch.full((4, 1), float("nan"), dtype=DTYPE))
    result = _diagnostic_hermite_output(run)
    assert torch.allclose(result.output_state[:, 0], query**3, rtol=0, atol=1e-15)
    assert torch.isnan(run.output_state).all()
