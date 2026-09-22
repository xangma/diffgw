"""Historical low-order torque regression; no scientific validation claim."""
import pytest
import torch
from dev.native_rom.eob.torch.radiation_reaction import EOBRadiationReaction

def test_normalized_torque_obeys_energy_balance_and_zero_momentum_is_finite():
    nu = torch.tensor(.19, dtype=torch.float64)
    zero = torch.zeros_like(nu)
    rr = EOBRadiationReaction(nu, zero, zero)
    omega = torch.tensor(.008, dtype=torch.float64)
    radial, torque = rr.radiation_reaction_forces(omega, zero, zero)
    torch.testing.assert_close(nu * omega * torque, -rr.energy_flux(omega))
    assert torch.isfinite(radial) and radial == 0


@pytest.mark.parametrize("samples", [2, 3])
def test_normalized_torque_batches_parameters_before_frequency(samples):
    nu = torch.tensor([.17, .24], dtype=torch.float64)
    rr = EOBRadiationReaction(nu, torch.zeros_like(nu), torch.zeros_like(nu))
    omega = torch.full((2, samples), .012, dtype=torch.float64)
    _, torque = rr.radiation_reaction_forces(omega, torch.zeros_like(omega), torch.ones_like(omega))
    torch.testing.assert_close(nu[:, None] * omega * torque, -rr.energy_flux(omega))
