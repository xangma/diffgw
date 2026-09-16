"""Unit tests for TaylorF2_HM multipolar frequency-domain approximant.

Formulas derived strictly from clean-room literature:
- Kidder (2008), Phys. Rev. D 77, 044016 (arXiv:0710.0614)
- Arun et al. (2009), Phys. Rev. D 79, 104023 (arXiv:0810.5336)
- Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
"""

import math

import pytest
import torch

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.taylorf2_hm import TaylorF2_HM
from torchwave.waveforms import get_fd_waveform


def test_taylorf2_hm_22_matches_taylorf2():
    """When only mode (2, 2) is active, TaylorF2_HM must match TaylorF2 to machine precision."""
    f = torch.linspace(30.0, 400.0, 200, dtype=torch.float64)
    m1, m2 = 15.0, 10.0
    chi1, chi2 = 0.3, -0.2
    iota = 0.7
    phic = 0.4

    hp_f2, hc_f2 = TaylorF2()(
        f, m1, m2, chi1=chi1, chi2=chi2, inclination=iota, phic=phic, f_isco_cutoff=True
    )
    hp_hm, hc_hm = TaylorF2_HM(modes=((2, 2),))(
        f, m1, m2, chi1=chi1, chi2=chi2, inclination=iota, phic=phic, f_isco_cutoff=True
    )

    diff_p = torch.max(torch.abs(hp_f2 - hp_hm)).item()
    diff_c = torch.max(torch.abs(hc_f2 - hc_hm)).item()
    scale = torch.max(torch.abs(hp_f2)).item()

    assert diff_p / scale < 1e-12
    assert diff_c / scale < 1e-12


def test_taylorf2_hm_odd_m_vanish_equal_mass():
    """For equal masses, the leading odd-m modes vanish identically."""
    f = torch.linspace(30.0, 300.0, 100, dtype=torch.float64)
    m1, m2 = 10.0, 10.0
    iota = 0.8

    hp, hc, modes = TaylorF2_HM()(
        f, m1, m2, inclination=iota, return_modes=True, f_isco_cutoff=False
    )

    hp_21, hc_21 = modes[(2, 1)]
    hp_31, hc_31 = modes[(3, 1)]
    hp_33, hc_33 = modes[(3, 3)]
    hp_43, hc_43 = modes[(4, 3)]

    assert torch.all(hp_21 == 0.0)
    assert torch.all(hc_21 == 0.0)
    assert torch.all(hp_31 == 0.0)
    assert torch.all(hc_31 == 0.0)
    assert torch.all(hp_33 == 0.0)
    assert torch.all(hc_33 == 0.0)
    assert torch.all(hp_43 == 0.0)
    assert torch.all(hc_43 == 0.0)

    # Even mode (4, 4) should NOT vanish
    hp_44, _ = modes[(4, 4)]
    assert torch.max(torch.abs(hp_44)).item() > 0.0

    # The even-m (3, 2) mode remains nonzero for equal, nonspinning masses.
    hp_32, _ = modes[(3, 2)]
    assert torch.max(torch.abs(hp_32)).item() > 0.0


def test_taylorf2_hm_32_leading_pn_ratio():
    """The added (3,2) mode follows the published leading PN amplitude."""
    f = torch.tensor([40.0, 80.0, 160.0], dtype=torch.float64)
    m1, m2 = 30.0, 10.0
    eta = m1 * m2 / (m1 + m2) ** 2
    v22 = (math.pi * (m1 + m2) * 4.925490947641267e-6 * f).pow(1.0 / 3.0)

    _, _, modes = TaylorF2_HM(modes=((2, 2), (3, 2)))(
        f,
        m1,
        m2,
        inclination=0.7,
        f_isco_cutoff=False,
        return_modes=True,
    )
    hp22, _ = modes[(2, 2)]
    hp32, _ = modes[(3, 2)]

    # Both modes have m=2 and therefore the same SPA phase here. Remove the
    # angular projection to compare the mode-amplitude ratio directly.
    from torchwave.approximants.taylorf2_hm import _swsh_projection_coefficients

    d22, _ = _swsh_projection_coefficients(2, 2, torch.tensor(0.7, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    d32, _ = _swsh_projection_coefficients(3, 2, torch.tensor(0.7, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    ratio = (hp32 / d32) / (hp22 / d22)
    # García-Quirós et al. (2020), Appendix E, Eq. (75), after matching the
    # common SPA normalization in Eq. (71): (1/3) sqrt(5/7)(1-3 eta) v^2.
    expected = (1.0 / 3.0) * math.sqrt(5.0 / 7.0) * (1.0 - 3.0 * eta) * v22.square()
    assert torch.allclose(ratio.real, expected, rtol=1e-13, atol=1e-15)
    assert torch.max(torch.abs(ratio.imag)).item() < 1e-13


def test_taylorf2_hm_forward_accepts_42_mode():
    """The per-call mode override accepts the published leading (4,2) term."""
    _, _, modes = TaylorF2_HM()(torch.tensor([40.0], dtype=torch.float64), 30.0, 10.0,
                                modes=((4, 2),), f_isco_cutoff=False,
                                return_modes=True)
    assert torch.isfinite(modes[(4, 2)][0]).all()
    assert torch.isfinite(modes[(4, 2)][1]).all()


def test_taylorf2_hm_unequal_mass_presence():
    """For unequal masses (q > 1), all modes have non-zero contribution."""
    f = torch.linspace(30.0, 300.0, 100, dtype=torch.float64)
    m1, m2 = 30.0, 10.0
    iota = 0.8

    hp, hc, modes = TaylorF2_HM()(
        f, m1, m2, inclination=iota, return_modes=True, f_isco_cutoff=False
    )

    for (l, m), (hp_m, hc_m) in modes.items():
        max_amp = torch.max(torch.abs(hp_m)).item()
        assert max_amp > 0.0, f"Mode ({l}, {m}) was zero for unequal masses"

    # Verify that total polarizations equal the sum over all modes
    hp_sum = sum(hp_m for hp_m, _ in modes.values())
    hc_sum = sum(hc_m for _, hc_m in modes.values())
    assert torch.allclose(hp, hp_sum, atol=1e-15)
    assert torch.allclose(hc, hc_sum, atol=1e-15)


def test_taylorf2_hm_mode_cutoffs():
    """Each harmonic cutoff must occur at f_cut,m = (m / 2) * f_ISCO."""
    f = torch.linspace(20.0, 1500.0, 1000, dtype=torch.float64)
    m1, m2 = 10.0, 5.0
    iota = 0.8

    hp, hc, modes = TaylorF2_HM()(
        f, m1, m2, inclination=iota, return_modes=True, f_isco_cutoff=True
    )

    f_cut_dict = {}
    for (l, m), (hp_m, _) in modes.items():
        nonzero_idx = torch.where(torch.abs(hp_m) > 0.0)[0]
        f_cut_dict[m] = f[nonzero_idx[-1]].item()

    # m=1 should cut off before m=2, before m=3, before m=4
    assert f_cut_dict[1] < f_cut_dict[2] < f_cut_dict[3] < f_cut_dict[4]


def test_taylorf2_hm_batching():
    """TaylorF2_HM must support batched inputs (B, N_f)."""
    f = torch.linspace(30.0, 400.0, 150, dtype=torch.float64)
    B = 4
    m1 = torch.tensor([10.0, 15.0, 20.0, 25.0], dtype=torch.float64)
    m2 = torch.tensor([5.0, 8.0, 10.0, 12.0], dtype=torch.float64)
    chi1 = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float64)
    chi2 = torch.tensor([-0.1, 0.2, -0.3, 0.4], dtype=torch.float64)
    inc = torch.tensor([0.2, 0.5, 0.8, 1.1], dtype=torch.float64)

    hp, hc = TaylorF2_HM()(f, m1, m2, chi1=chi1, chi2=chi2, inclination=inc)
    assert hp.shape == (B, len(f))
    assert hc.shape == (B, len(f))
    assert not torch.isnan(hp).any()
    assert not torch.isnan(hc).any()


def test_taylorf2_hm_autograd():
    """Autograd gradients must propagate cleanly with zero NaNs."""
    f = torch.linspace(30.0, 300.0, 50, dtype=torch.float64)
    m1 = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)

    hp, hc = TaylorF2_HM()(f, m1, m2, chi1=chi1, chi2=chi2, inclination=inc, f_isco_cutoff=False)
    loss = (hp.real.sum() + hc.imag.sum())
    loss.backward()

    for name, param in [("m1", m1), ("m2", m2), ("chi1", chi1), ("chi2", chi2), ("inc", inc)]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad).any(), f"Gradient for {name} has NaNs"


def test_get_fd_waveform_taylorf2_hm():
    """Verify high-level get_fd_waveform API with TaylorF2_HM."""
    f = torch.linspace(30.0, 300.0, 100, dtype=torch.float64)
    hp, hc = get_fd_waveform(
        approximant="TaylorF2_HM",
        mass1=12.0,
        mass2=8.0,
        chi1=0.1,
        chi2=-0.1,
        sample_frequencies=f,
    )
    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.max(torch.abs(hp)).item() > 0.0
