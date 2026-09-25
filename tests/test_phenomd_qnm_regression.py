"""Independent Kerr eigenproblem, context, and derivative regressions."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from torchwave import get_fd_waveform
from torchwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220


def _validation():
    return json.loads((Path(__file__).parent / "data/kerr_qnm_220_leaver.json").read_text())


def test_qnm_matches_independent_heldout_eigenproblem_solves():
    rows = _validation()["holdouts"]
    spin = torch.tensor([r["a"] for r in rows], dtype=torch.float64)
    expected = torch.tensor([[r["wr"], r["damping"]] for r in rows], dtype=torch.float64)
    actual = torch.stack(evaluate_kerr_qnm_220(spin), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1.1e-10, atol=0)


def test_qnm_matches_stricter_independent_solves():
    # Higher angular truncation and tighter continued fractions at difficult spins.
    rows = _validation()["strict"]
    spin = torch.tensor([r["a"] for r in rows], dtype=torch.float64)
    expected = torch.tensor([[r["wr"], r["damping"]] for r in rows], dtype=torch.float64)
    actual = torch.stack(evaluate_kerr_qnm_220(spin), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1.1e-10, atol=0)


def test_qnm_known_author_table_outlier_is_not_reproduced():
    # Lo et al. 2025 data gives damping 0.0160559974464903 at a=.9976.
    # Independent tighter eigenproblem solves give 0.0157961469550101.
    _, damping = evaluate_kerr_qnm_220(torch.tensor(0.9976, dtype=torch.float64))
    assert abs(damping.item() / 0.0157961469550101 - 1) < 1e-10
    assert abs(damping.item() / 0.0160559974464903 - 1) > 0.01


def test_qnm_inference_then_autograd_without_data_reads(monkeypatch):
    def no_file_reads(*args, **kwargs):
        raise AssertionError("QNM runtime must not load a file")

    monkeypatch.setattr(Path, "open", no_file_reads)
    with torch.inference_mode():
        evaluate_kerr_qnm_220(torch.tensor(0.4, dtype=torch.float64))
    spin = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    wr, damping = evaluate_kerr_qnm_220(spin)
    assert torch.autograd.grad(wr, spin, retain_graph=True)[0] > 0
    assert torch.autograd.grad(damping, spin)[0] < 0


@pytest.mark.parametrize("evaluate", [evaluate_kerr_qnm_220, signed_qnm_220])
def test_qnm_zero_spin_derivative_matches_independent_roots(evaluate):
    rows = {r["a"]: r for r in _validation()["holdouts"]}
    spin = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    for name, value in zip(("wr", "damping"), evaluate(spin)):
        slope = (rows[1e-7][name] - rows[-1e-7][name]) / 2e-7
        gradient = torch.autograd.grad(value, spin, retain_graph=True)[0]
        assert abs(gradient.item() - slope) < 5e-8


def test_qnm_clamps_to_validated_domain():
    values = torch.tensor([-1.2, -0.9999, 0.9999, 1.2], dtype=torch.float64)
    for channel in evaluate_kerr_qnm_220(values):
        assert channel[0] == channel[1]
        assert channel[2] == channel[3]


@pytest.mark.parametrize("spin", [-0.9995, -0.85317, -1e-7, 0.0, 1e-7, 0.41327, 0.9995])
def test_qnm_gradients_match_finite_differences(spin):
    # Smaller steps near extremality resolve curvature; no parity tolerance is changed.
    value = torch.tensor(spin, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(evaluate_kerr_qnm_220, (value,), eps=1e-7, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_qnm_batches_and_execution_contexts(dtype):
    spin = torch.tensor([[-0.99, -0.2], [0.0, 0.99]], dtype=dtype)
    normal = evaluate_kerr_qnm_220(spin)
    with torch.no_grad():
        no_grad = evaluate_kerr_qnm_220(spin)
    with torch.inference_mode():
        inference = evaluate_kerr_qnm_220(spin)
    for i in range(2):
        assert normal[i].shape == spin.shape
        assert normal[i].dtype == dtype
        assert torch.equal(normal[i], no_grad[i])
        assert torch.equal(normal[i], inference[i])
        scalar = torch.stack([evaluate_kerr_qnm_220(s)[i] for s in spin.flatten()]).reshape_as(spin)
        torch.testing.assert_close(normal[i], scalar)


def test_phenomd_waveform_spin_gradient_and_polarization_scaling():
    frequency = torch.linspace(20.0, 400.0, 321, dtype=torch.float64)

    def waveform(spin):
        return get_fd_waveform(
            "IMRPhenomD", mass1=30.0, mass2=20.0, chi1=spin, chi2=-0.2,
            distance=400.0, inclination=0.5, phic=0.3, f_ref=20.0,
            sample_frequencies=frequency,
        )

    spin = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    hp, hc = waveform(spin)
    scale = torch.linalg.vector_norm(hp).detach()
    assert torch.isfinite(hp).all() and scale > 0
    ratio = -2j * np.cos(0.5) / (1 + np.cos(0.5) ** 2)
    assert torch.linalg.vector_norm(hc - ratio * hp) / scale < 1e-12
    objective = (hp.real / scale).sum()
    actual = torch.autograd.grad(objective, spin)[0]
    step = 1e-6
    hi, _ = waveform(torch.tensor(0.3 + step, dtype=torch.float64))
    lo, _ = waveform(torch.tensor(0.3 - step, dtype=torch.float64))
    expected = ((hi.real - lo.real) / (2 * step * scale)).sum()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
