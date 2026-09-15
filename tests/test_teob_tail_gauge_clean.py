"""Focused tests for the TEOB-only tail phase gauge correction."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from torchwave.approximants.eob.teob_radiation_clean import (
    TEOBRadiationClean,
    teob_tail_gauge_factor,
)
from torchwave.approximants.eob.teob_ringdown_assembly_clean import (
    natural_cubic_interpolate,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian
from torchwave.constants import C_SI, MPC_SI


_FIXTURE = Path("/tmp/torchwave-teob-pa-public-gauge.npz")
_FIXTURE_SHA256 = "c81bcb89086833fa6c44fc44a26eb54ee93adc2ff14fca5976944aaae54c8fb0"


def _load_fixture() -> np.lib.npyio.NpzFile:
    if not _FIXTURE.is_file():
        pytest.skip(f"public PA gauge fixture is unavailable: {_FIXTURE}")
    digest = hashlib.sha256(_FIXTURE.read_bytes()).hexdigest()
    assert digest == _FIXTURE_SHA256, "public PA gauge fixture hash changed"
    return np.load(_FIXTURE, allow_pickle=False)


def _evaluate_fixture_mode(data: np.lib.npyio.NpzFile) -> tuple[torch.Tensor, TEOBRadiationClean]:
    state = torch.from_numpy(data["pa_state"]).to(torch.float64)
    radius, phase, prstar, pphi = state.unbind(-1)
    omega = torch.from_numpy(data["Omega"]).to(torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    hamiltonian = TEOBResumSHamiltonian(
        torch.tensor(0.24, dtype=torch.float64), zero, zero, zero, zero,
        delta=torch.tensor(0.2, dtype=torch.float64),
    )
    heff, hreal, hcirc, aprime = hamiltonian.clean_radiation_inputs(
        radius, prstar, pphi,
    )
    radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")
    mode = radiation.mode(
        2, 2, radius, omega, phase, prstar, pphi, heff, hreal, hcirc,
        aprime, hamiltonian.m1, hamiltonian.m2,
    )
    torch.testing.assert_close(
        hreal, torch.from_numpy(data["Hreal"]).to(torch.float64),
        rtol=0.0, atol=2e-15,
    )
    return mode, radiation


def test_teob_tail_gauge_factor_has_published_phase_and_finite_gradients():
    real_energy = torch.tensor([0.91, 1.07], dtype=torch.float64, requires_grad=True)
    orbital_frequency = torch.tensor([0.013, 0.021], dtype=torch.float64, requires_grad=True)
    factor = teob_tail_gauge_factor(2, real_energy, orbital_frequency)
    angle = -2.0 * real_energy * orbital_frequency
    expected = torch.complex(torch.cos(angle), torch.sin(angle))
    torch.testing.assert_close(factor, expected, rtol=0.0, atol=2e-15)
    torch.testing.assert_close(factor.abs(), torch.ones_like(real_energy), rtol=0.0, atol=2e-15)

    objective = factor.real.sum() + 0.25 * factor.imag.sum()
    gradients = torch.autograd.grad(objective, (real_energy, orbital_frequency))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(bool(torch.any(gradient != 0)) for gradient in gradients)


def test_teob_tail_gauge_corrects_public_pa_mode_without_changing_amplitude_or_flux():
    data = _load_fixture()
    mode, radiation = _evaluate_fixture_mode(data)
    before = torch.from_numpy(data["mode_before"])
    hreal = torch.from_numpy(data["Hreal"]).to(torch.float64)
    omega = torch.from_numpy(data["Omega"]).to(torch.float64)

    expected = before * teob_tail_gauge_factor(2, hreal, omega)
    torch.testing.assert_close(mode, expected, rtol=2e-13, atol=2e-15)
    torch.testing.assert_close(mode.abs(), before.abs(), rtol=0.0, atol=2e-15)

    state = torch.from_numpy(data["pa_state"]).to(torch.float64)
    radius, phase, prstar, pphi = state.unbind(-1)
    zero = torch.tensor(0.0, dtype=torch.float64)
    hamiltonian = TEOBResumSHamiltonian(
        torch.tensor(0.24, dtype=torch.float64), zero, zero, zero, zero,
        delta=torch.tensor(0.2, dtype=torch.float64),
    )
    heff, hreal_eval, hcirc, aprime = hamiltonian.clean_radiation_inputs(
        radius, prstar, pphi,
    )
    amplitudes = radiation.amplitudes(
        radius, omega, phase, prstar, pphi, heff, hreal_eval, hcirc, aprime,
        hamiltonian.m1, hamiltonian.m2,
    )
    torch.testing.assert_close(amplitudes[(2, 2)], before.abs(), rtol=2e-13, atol=2e-15)
    infinity_flux = radiation.flux(
        radius, omega, phase, prstar, pphi, heff, hreal_eval, hcirc, aprime,
        hamiltonian.m1, hamiltonian.m2,
    )
    mode_flux = sum(
        m * m * omega * value.square() for (_, m), value in amplitudes.items()
    ) / (8.0 * math.pi)
    torch.testing.assert_close(infinity_flux, mode_flux, rtol=2e-13, atol=2e-15)


def test_teob_tail_gauge_matches_public_projected_pa_strain():
    data = _load_fixture()
    mode, _ = _evaluate_fixture_mode(data)
    public_time = torch.from_numpy(data["public_time"]).to(torch.float64)
    pa_time = torch.from_numpy(data["pa_time"]).to(torch.float64)
    state = torch.from_numpy(data["pa_state"]).to(torch.float64)
    orbital_phase = state[:, 1]
    valid = (public_time >= pa_time[0]) & (public_time <= pa_time[-1])
    query = public_time[valid]

    # Lift the residual phase after removing the orbital carrier at the PA
    # knots.  Interpolate amplitude and positive phase independently onto the
    # public grid; the public complex strain stays at its native dense times.
    phase_knots = 2.0 * orbital_phase - torch.from_numpy(
        np.unwrap(np.angle(
            mode.detach().numpy() * np.exp(2j * orbital_phase.numpy())
        ))
    ).to(torch.float64)
    amplitude = natural_cubic_interpolate(pa_time, mode.abs(), query)
    phase = natural_cubic_interpolate(pa_time, phase_knots, query)
    mode_dense = torch.polar(amplitude, -phase).detach().numpy()

    public = data["public_hp"][valid.numpy()] - 1j * data["public_hc"][valid.numpy()]
    scale = -math.sqrt(5.0 / (4.0 * math.pi)) * float(data["Msec"]) * C_SI / (100.0 * MPC_SI)
    projected = scale * mode_dense
    residual = public / projected
    phase_residual = np.unwrap(np.angle(residual))
    relative_l2 = np.linalg.norm(public - projected) / np.linalg.norm(public)
    public_amplitude = np.abs(public)
    projected_amplitude = np.abs(projected)
    amplitude_relative_l2 = (
        np.linalg.norm(projected_amplitude - public_amplitude)
        / np.linalg.norm(public_amplitude)
    )

    # This bounded dense-grid interpolation comparison is regression evidence,
    # with tolerances that reject the approximately 2.2e-2-rad old-gauge offset.
    assert abs(phase_residual[0]) < 2e-6
    assert np.max(np.abs(phase_residual)) < 4e-4
    assert relative_l2 < 2e-4
    assert amplitude_relative_l2 < 2e-6
