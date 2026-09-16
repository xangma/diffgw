"""TEOBResumS factorized radiation-reaction contract.

Nagar et al., arXiv:1806.01772v2, Sec. II states that the azimuthal force
uses the factorized multipolar waveform flux and that the radial force is
set to zero.  These tests exercise that public equation against the
independently reconstructed DIN mode sum; they do not use target-library
implementation details or outputs.
"""

import torch

from torchwave.approximants.eob.factorized_modes import FactorizedModes
from torchwave.approximants.teobresums import (
    TEOBResumSHamiltonian,
    TEOBResumSRadiationReaction,
)


def test_teob_factorized_torque_is_din_flux_over_nu_omega():
    dtype = torch.float64
    nu = torch.tensor(0.25, dtype=dtype)
    hamiltonian = TEOBResumSHamiltonian(
        nu, torch.zeros_like(nu), torch.zeros_like(nu),
        torch.tensor(10.0, dtype=dtype), torch.zeros_like(nu),
    )
    rr = TEOBResumSRadiationReaction(
        hamiltonian.nu, hamiltonian.chi1, hamiltonian.chi2,
    )
    omega = torch.tensor([0.015, 0.025], dtype=dtype)
    radius = torch.tensor([12.0, 9.0], dtype=dtype)
    radial_momentum = torch.tensor([-0.01, -0.02], dtype=dtype)
    angular_momentum = torch.tensor([3.4, 3.1], dtype=dtype)

    radial, torque = rr.radiation_reaction_forces(
        omega, radial_momentum, angular_momentum,
        r=radius, hamiltonian=hamiltonian,
    )
    expected_flux = rr._factorized.energy_flux(
        omega, radial_momentum, angular_momentum,
        r=radius, hamiltonian=hamiltonian,
    )

    torch.testing.assert_close(radial, torch.zeros_like(radial))
    torch.testing.assert_close(
        torque, -expected_flux / (hamiltonian.nu * omega),
        rtol=2e-14, atol=2e-14,
    )
    assert torch.all(expected_flux > 0)
    assert torch.all(torch.isfinite(expected_flux))


def test_teob_factorized_path_uses_all_published_din_modes():
    rr = TEOBResumSRadiationReaction(
        torch.tensor(0.24, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert isinstance(rr._factorized.modes, FactorizedModes)
    assert rr._factorized.modes.lmax == 8
    assert rr._factorized.modes.spin_model == "none"
