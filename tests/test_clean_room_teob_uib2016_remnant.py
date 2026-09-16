"""Direct tests for the published nonspinning UIB2016 TEOB remnant branch."""

import hashlib
import os
import math
from pathlib import Path

import pytest
import torch

from torchwave.approximants.eob.teob_uib2016_remnant import (
    uib2016_nonspinning_final_mass_fraction,
    uib2016_nonspinning_final_spin,
    uib2016_nonspinning_radiated_energy,
    uib2016_nonspinning_remnant,
)


def test_full_precision_author_asset_and_paper_equations_at_equal_mass():
    # Independent arithmetic from Jimenez-Forteza Eqs. (7) and (21), with
    # full-precision coefficients printed in the author IMRPhenomXAS asset.
    eta = torch.tensor(0.25, dtype=torch.float64)
    expected_energy = (
        (1.0 - 2.0 * math.sqrt(2.0) / 3.0) * 0.25
        + 0.5609904135313374 * 0.25**2
        - 0.84667563764404 * 0.25**3
        + 3.145145224278187 * 0.25**4
    )
    expected_spin = (
        2.0 * math.sqrt(3.0) * 0.25
        + (5.24 * 3.8326341618708577) * 0.25**2
        + (1.3 * -9.487364155598392) * 0.25**3
    ) / (1.0 + (2.88 * 2.5134875145648374) * 0.25)
    torch.testing.assert_close(
        uib2016_nonspinning_radiated_energy(eta),
        torch.tensor(expected_energy, dtype=torch.float64),
        rtol=2e-15,
        atol=2e-16,
    )
    torch.testing.assert_close(
        uib2016_nonspinning_final_mass_fraction(eta),
        torch.tensor(1.0 - expected_energy, dtype=torch.float64),
        rtol=2e-15,
        atol=2e-16,
    )
    torch.testing.assert_close(
        uib2016_nonspinning_final_spin(eta),
        torch.tensor(expected_spin, dtype=torch.float64),
        rtol=2e-15,
        atol=2e-16,
    )


def test_test_mass_limits_are_the_published_linear_terms():
    eta = torch.tensor([1e-8, 1e-7], dtype=torch.float64)
    energy_over_eta = uib2016_nonspinning_radiated_energy(eta) / eta
    spin_over_eta = uib2016_nonspinning_final_spin(eta) / eta
    torch.testing.assert_close(
        energy_over_eta,
        torch.full_like(eta, 1.0 - 2.0 * math.sqrt(2.0) / 3.0),
        rtol=2e-6,
        atol=0,
    )
    torch.testing.assert_close(
        spin_over_eta,
        torch.full_like(eta, 2.0 * math.sqrt(3.0)),
        rtol=2e-6,
        atol=0,
    )
    torch.testing.assert_close(
        uib2016_nonspinning_final_mass_fraction(torch.tensor(0.0)),
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        uib2016_nonspinning_final_spin(torch.tensor(0.0)),
        torch.tensor(0.0),
    )


def test_component_mass_batching_and_scaling():
    mass1 = torch.tensor([[2.0], [10.0]], dtype=torch.float64)
    mass2 = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
    mf, spin = uib2016_nonspinning_remnant(mass1, mass2)
    assert mf.shape == (2, 3)
    assert spin.shape == (2, 3)
    for i in range(mass1.shape[0]):
        for j in range(mass2.shape[0]):
            scalar = uib2016_nonspinning_remnant(mass1[i, 0], mass2[j])
            torch.testing.assert_close(mf[i, j], scalar[0])
            torch.testing.assert_close(spin[i, j], scalar[1])
    # The symmetric mass ratio is invariant under a common mass scale.
    scaled = uib2016_nonspinning_remnant(1e200 * mass1, 1e200 * mass2)
    torch.testing.assert_close(mf, scaled[0], rtol=1e-14, atol=0)
    torch.testing.assert_close(spin, scaled[1], rtol=1e-14, atol=0)


def test_nearly_equal_masses_are_not_rejected_by_eta_roundoff():
    mf, spin = uib2016_nonspinning_remnant(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(1.0 + 1e-8, dtype=torch.float64),
    )
    torch.testing.assert_close(mf, uib2016_nonspinning_final_mass_fraction(0.25))
    torch.testing.assert_close(spin, uib2016_nonspinning_final_spin(0.25))


def test_boolean_tensors_are_not_accepted_as_real_inputs():
    with pytest.raises(TypeError):
        uib2016_nonspinning_radiated_energy(torch.tensor(False))


def test_remnant_gradients_are_finite_and_nonzero():
    masses = torch.tensor([7.0, 1.4], dtype=torch.float64, requires_grad=True)
    mf, spin = uib2016_nonspinning_remnant(*masses)
    gradients = torch.autograd.grad(mf + spin, masses)
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all((value.abs() > 0).all() for value in gradients)


@pytest.mark.parametrize(
    "call",
    [
        lambda: uib2016_nonspinning_radiated_energy(torch.tensor(float("nan"))),
        lambda: uib2016_nonspinning_final_spin(torch.tensor(0.250001)),
        lambda: uib2016_nonspinning_remnant(0.0, 1.4),
        lambda: uib2016_nonspinning_remnant(float("inf"), 1.4),
    ],
)
def test_invalid_domain_is_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_author_asset_hash_is_pinned_for_provenance():
    location = os.environ.get("TORCHWAVE_EXTERNAL_XAS_ASSET")
    if not location:
        pytest.skip("XAS author asset requires explicit external opt-in")
    asset = Path(location)
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    assert digest == "94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52"
