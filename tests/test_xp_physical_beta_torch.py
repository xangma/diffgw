"""Production zero-nutation opening against the independent scalar identity."""

import math

import pytest
import torch

from diffgw_reference.xp_msa_regularized_beta import (
    aligned_minimal_rotation_gamma, one_spin_aligned_beta,
)
import torchwave.approximants.precession_msa_waveform_clean as waveform_clean
from torchwave.approximants.precession_msa_reference_l_clean import (
    orbital_angular_momentum_nonconserved_3pn,
)
from torchwave.approximants.precession_msa_waveform_clean import (
    physical_zero_nutation_cos_beta,
)


@pytest.mark.parametrize("mass1,mass2", ((40.0, 20.0), (30.0, 30.0)))
@pytest.mark.parametrize("epsilon,azimuth", ((0.0, 0.0), (1e-6, 0.7), (0.3, -1.4)))
def test_torch_opening_matches_scalar_reference(mass1, mass2, epsilon, azimuth):
    frequency = torch.tensor([20.0, 40.0, 100.0], dtype=torch.float64)
    spin1 = torch.tensor(
        [epsilon * math.cos(azimuth), epsilon * math.sin(azimuth), 0.4],
        dtype=torch.float64,
    )
    spin2 = torch.tensor([0.0, 0.0, -0.2], dtype=torch.float64)
    actual = physical_zero_nutation_cos_beta(
        frequency, torch.tensor(mass1, dtype=torch.float64),
        torch.tensor(mass2, dtype=torch.float64), spin1, spin2,
    )
    orbital_l = orbital_angular_momentum_nonconserved_3pn(
        mass1, mass2, frequency, spin1[2], spin2[2],
    )
    expected = torch.tensor([
        one_spin_aligned_beta(
            mass1, mass2, tuple(spin1.tolist()), tuple(spin2.tolist()),
            v_ref=0.2, l_beta=float(value),
        ).cos_beta
        for value in orbital_l
    ], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)


def test_torch_opening_has_finite_endpoint_and_interior_gradients():
    frequency = torch.tensor([20.0, 40.0, 100.0], dtype=torch.float64)
    for transverse in (0.0, 0.3):
        sx = torch.tensor(transverse, dtype=torch.float64, requires_grad=True)
        sz = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        spin1 = torch.stack((sx, torch.zeros_like(sx), sz))
        spin2 = torch.tensor([0.0, 0.0, -0.2], dtype=torch.float64)
        cosine = physical_zero_nutation_cos_beta(
            frequency, torch.tensor(40.0), torch.tensor(20.0), spin1, spin2,
        )
        gradients = torch.autograd.grad(cosine.sum(), (sx, sz))
        assert all(torch.isfinite(value) for value in gradients)
        if transverse:
            assert gradients[0] < 0


def test_torch_aligned_transport_obeys_scalar_minimal_rotation(monkeypatch):
    frequency = torch.tensor([25.0, 40.0, 80.0, 300.0], dtype=torch.float64)
    captured = {}
    original = waveform_clean.compose_msa_waveform

    def record(*args, **kwargs):
        captured.update(args[5])
        return original(*args, **kwargs)

    monkeypatch.setattr(waveform_clean, "compose_msa_waveform", record)
    waveform_clean.waveform_v223(
        frequency, 36.0, 22.0, spin1z=0.35, spin2z=-0.25,
        distance=150.0, inclination=0.65, phic=0.0, f_ref=40.0,
        convention=0, boundary="physical",
    )
    expected = torch.tensor([
        aligned_minimal_rotation_gamma(float(value))
        for value in captured["alpha"]
    ], dtype=torch.float64)
    torch.testing.assert_close(captured["gamma"], expected, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(captured["cos_beta"], torch.ones_like(frequency))
