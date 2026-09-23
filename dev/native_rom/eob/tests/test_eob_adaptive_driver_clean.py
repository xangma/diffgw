import inspect

import torch
import pytest

from dev.native_rom.eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive


DTYPE = torch.float64


def test_adaptive_compatibility_defaults_are_explicitly_reviewed_choices():
    signature = inspect.signature(reconstruct_seobnrv4_22_adaptive)
    assert signature.parameters["spin_truncation"].default == "generic-leading-ell8"
    assert signature.parameters["final_spin_fit_order"].default == (3, 4)
    omega = torch.tensor(0.05, dtype=DTYPE)
    with pytest.raises(ValueError, match="spin_truncation"):
        reconstruct_seobnrv4_22_adaptive(omega, 30.0, 30.0,
                                          spin_truncation="tabulated")
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
