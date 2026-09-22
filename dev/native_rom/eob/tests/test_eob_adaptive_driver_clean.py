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
