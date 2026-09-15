"""Verify own equation-solved frequencies against frozen external Cook rows.

These external data are never used by our generator or runtime.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from torchwave.approximants.kerr_qnm_author import (
    SPIN_MAX, SUPPORTED_MODES, dimensionless_qnm,
)

EXTERNAL_DATA = Path(os.environ["TORCHWAVE_EXTERNAL_VALIDATION_DIR"]) if os.environ.get("TORCHWAVE_EXTERNAL_VALIDATION_DIR") else None
if EXTERNAL_DATA is not None:
    HOLDOUT = json.loads((EXTERNAL_DATA / "kerr_qnm_author_holdout.json").read_text())
    VALIDATION = json.loads((EXTERNAL_DATA / "kerr_qnm_author_validation.json").read_text())
else:
    HOLDOUT = VALIDATION = {"modes": {}}
AUTHOR_MODES = ([tuple(map(int, label.split(","))) for label in HOLDOUT["modes"]]
                if EXTERNAL_DATA is not None else SUPPORTED_MODES)


@pytest.mark.parametrize("mode", AUTHOR_MODES)
def test_frequencies_against_disjoint_author_holdouts(mode):
    if EXTERNAL_DATA is None:
        pytest.skip("Cook-derived validation data require explicit opt-in")
    label = ",".join(map(str, mode))
    fixture = HOLDOUT["modes"][label]
    report = VALIDATION["modes"][label]
    assert not set(fixture["row_indices"]) & set(report["training_row_indices"])
    rows = torch.tensor(fixture["rows"], dtype=torch.float64)
    real, decay = dimensionless_qnm(*mode, rows[:, 0])
    error = torch.stack((abs(real - rows[:, 1]), abs(-decay - rows[:, 2])), dim=-1)
    # The archived measured envelope belongs to the retired Cook-derived
    # interpolant. It cannot bound a different interpolant of our own roots.
    # Retain the existing external 5e-10 bound and extend it to all these spins;
    # our stricter 1e-12 AND 1e-10-relative gates use fresh equation solves.
    assert torch.all(error <= 5e-10)
    assert torch.all(real > 0) and torch.all(decay > 0)


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_spin_derivative_through_legacy_compatibility_import(mode):
    spin = torch.tensor([.17, .731, .962, .99125], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: dimensionless_qnm(*mode, a),
                                    (spin,), eps=1e-8, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_batch_shape_and_domain_endpoints(dtype):
    spin = torch.tensor([[0., .3], [.8, SPIN_MAX]], dtype=dtype, requires_grad=True)
    real, decay = dimensionless_qnm(2, 2, 0, spin)
    assert real.shape == decay.shape == spin.shape
    assert real.dtype == decay.dtype == dtype
    assert real.device == decay.device == spin.device
    assert torch.all(torch.isfinite(real)) and torch.all(torch.isfinite(decay))
    assert torch.all(decay > 0)
    gradient, = torch.autograd.grad((real + decay).sum(), spin)
    assert torch.all(torch.isfinite(gradient))


@pytest.mark.parametrize("spin", [-.1, 1., float("nan"), float("inf")])
def test_no_unmeasured_extrapolation(spin):
    with pytest.raises(ValueError, match="spin domain"):
        dimensionless_qnm(2, 2, 0, spin)


def test_rejects_unknown_mode_and_nonfloating_tensor():
    with pytest.raises(ValueError, match="Unsupported"):
        dimensionless_qnm(2, 2, 3, .7)
    with pytest.raises(TypeError, match="floating"):
        dimensionless_qnm(2, 2, 0, torch.tensor(0))


def test_schwarzschild_frequency_and_decay_sign():
    real, decay = dimensionless_qnm(2, 2, 0, 0.)
    assert real.dtype == torch.float64
    assert abs(real.item() - .3736716844180418) < 2e-15
    assert abs(decay.item() - .0889623156889357) < 2e-15
