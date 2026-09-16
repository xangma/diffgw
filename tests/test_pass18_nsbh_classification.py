"""Source-derived tests for the NSBH remnant frequency classification path."""

import torch

from torchwave.approximants.imrphenom_nsbh import (
    IMRPhenomNSBH,
    bbh_remnant_uib2016,
    london_ringdown_properties,
    nsbh_frequencies,
)


def test_classification_boundary_uses_nsbh_remnant_ringdown():
    """The Thompson table compares f_tide with the Zappa NSBH f_RD.

    This configuration lies between the NSBH and BBH ringdown frequencies;
    using the BBH conditional branch therefore changes the merger class.
    """
    dtype = torch.float64
    mass1 = torch.tensor(8.0, dtype=dtype)
    mass2 = torch.tensor(1.4, dtype=dtype)
    chi1 = torch.tensor(0.0, dtype=dtype)
    lambda2 = torch.tensor(1800.0, dtype=dtype)

    f_tide, _, f_rd_nsbh, _ = nsbh_frequencies(mass1, mass2, chi1, lambda2)
    mf_bbh_ratio, chi_bbh = bbh_remnant_uib2016(
        mass1, mass2, chi1, torch.zeros_like(chi1)
    )
    f_rd_bbh, _ = london_ringdown_properties(
        (mass1 + mass2) * mf_bbh_ratio, chi_bbh
    )

    # Eq. (table-merger-dep-quants): f_tide >= f_RD is the closed boundary.
    assert f_tide >= f_rd_nsbh
    assert f_tide < f_rd_bbh


def test_classification_frequency_batch_and_waveform_gradients():
    """The NSBH remnant path remains batched and differentiable away from a boundary."""
    dtype = torch.float64
    mass1 = torch.tensor([5.6, 4.2, 8.4], dtype=dtype, requires_grad=True)
    mass2 = torch.tensor([1.4, 1.4, 1.4], dtype=dtype, requires_grad=True)
    chi1 = torch.tensor([0.0, 0.3, -0.2], dtype=dtype, requires_grad=True)
    lambda2 = torch.tensor([800.0, 500.0, 300.0], dtype=dtype, requires_grad=True)

    f_tide, _, f_rd, _ = nsbh_frequencies(mass1, mass2, chi1, lambda2)
    assert f_tide.shape == f_rd.shape == (3,)
    assert torch.all(torch.isfinite(f_rd))

    f = torch.linspace(40.0, 500.0, 24, dtype=dtype)
    hp, hc = IMRPhenomNSBH()(
        f=f,
        mass1=mass1,
        mass2=mass2,
        chi1=chi1,
        lambda2=lambda2,
    )
    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()
    for value in (mass1, mass2, chi1, lambda2):
        assert value.grad is not None
        assert torch.all(torch.isfinite(value.grad))
