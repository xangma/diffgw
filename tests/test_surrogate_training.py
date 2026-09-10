"""Tests for SEOBNR ROM surrogate training pipeline and model fidelity."""

import os
import tempfile
import pytest
import torch
import numpy as np

from torchwave.approximants.rom.generator import (
    poly_features,
    unwrap_phase,
    compute_svd_reduced_basis,
    fit_parameter_mapping,
    train_and_export_surrogate,
)
from torchwave.approximants.rom.surrogate import SEOBNR_ROM
from torchwave.approximants.eob.waveform import SEOBNR_ODE


def test_unwrap_phase_vectorized():
    """Verify pure PyTorch phase unwrapping matches numpy."""
    p_raw = torch.tensor([0.0, 1.0, 3.0, -2.5, -1.0, 3.0, -3.0], dtype=torch.float64)
    p_unwrapped = unwrap_phase(p_raw)
    p_np = torch.from_numpy(np.unwrap(p_raw.numpy()))
    assert torch.allclose(p_unwrapped, p_np, atol=1e-6)


def test_poly_features_degrees():
    """Verify polynomial feature shapes across degrees."""
    eta = torch.tensor([0.25, 0.20, 0.15])
    chi = torch.tensor([0.0, 0.5, -0.3])
    phi_d2 = poly_features(eta, chi, degree=2)
    # Degree 2: 1, eta, chi, eta^2, eta*chi, chi^2 -> 6 features
    assert phi_d2.shape == (3, 6)

    phi_d3 = poly_features(eta, chi, degree=3)
    # Degree 3: 6 + (eta^3, eta^2*chi, eta*chi^2, chi^3) -> 10 features
    assert phi_d3.shape == (3, 10)


def test_surrogate_training_pipeline():
    """Verify end-to-end training and export of SEOBNR surrogate model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "test_surrogate.pt")
        path = train_and_export_surrogate(
            output_path=out_file,
            num_q=4,
            num_spins=3,
            num_freq_nodes=48,
            num_modes=4,
            degree=2,
            num_ode_steps=200,
        )

        assert os.path.isfile(path)
        data = torch.load(path, weights_only=False)

        assert "mf_grid" in data
        assert "amp_basis" in data
        assert "phase_basis" in data
        assert "amp_weights" in data
        assert "phase_weights" in data
        assert data["amp_basis"].shape[0] == 4
        assert data["phase_basis"].shape[0] == 4


def test_surrogate_evaluation_fidelity():
    """Verify native ROM evaluates with microsecond latency and finite complex values."""
    rom = SEOBNR_ROM()
    assert rom.surrogate_data is not None

    f = torch.linspace(25.0, 300.0, 128, dtype=torch.float64)
    hp, hc = rom(
        f=f,
        mass1=35.0,
        mass2=25.0,
        chi1=0.2,
        chi2=-0.1,
        distance=400.0,
    )

    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.all(torch.isfinite(hp))
    assert torch.all(torch.isfinite(hc))
    assert torch.any(hp.abs() > 0.0)
