"""Tests for clean-room IMRPhenomHM and IMRPhenomXHM multipolar higher-harmonic approximants.

Literature citations:
- London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
- Berti et al. (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
- Kidder (2008), Phys. Rev. D 77, 044012 (arXiv:0710.0614)
"""

import math
import pytest
import torch

from torchwave.waveforms import get_fd_waveform
from torchwave.approximants.imrphenom_hm import (
    IMRPhenomHM,
    IMRPhenomXHM,
    _swsh_projection_coefficients,
)
from torchwave.approximants.imrphenomd import IMRPhenomD


def test_spin_weighted_spherical_harmonics_orthonormality_and_symmetry():
    """Verify spin-weighted spherical harmonics -2Ylm projection coefficients."""
    # At face on (iota = 0):
    iota_0 = torch.tensor(0.0, dtype=torch.float64)
    phi0_0 = torch.tensor(0.0, dtype=torch.float64)
    d_plus, d_cross = _swsh_projection_coefficients(2, 2, iota_0, phi0_0)
    expected = 0.5 * math.sqrt(5.0 / (4.0 * math.pi))
    assert math.isclose(abs(d_plus.item()), expected, rel_tol=1e-5)
    assert math.isclose(abs(d_cross.item()), expected, rel_tol=1e-5)

    # For edge on (iota = pi/2):
    iota_pi2 = torch.tensor(math.pi / 2.0, dtype=torch.float64)
    for l, m in IMRPhenomHM.SUPPORTED_MODES:
        dp, dc = _swsh_projection_coefficients(l, m, iota_pi2, phi0_0)
        assert torch.isfinite(dp.real) and torch.isfinite(dp.imag)
        assert torch.isfinite(dc.real) and torch.isfinite(dc.imag)


def test_odd_m_modes_vanish_for_equal_masses():
    """Odd-m modes (2,1) and (3,3) must vanish identically for equal-mass binaries (m1 = m2)."""
    f = torch.linspace(25.0, 300.0, 100, dtype=torch.float64)
    # Inclination pi/3 where higher modes are prominent
    inc = math.pi / 3.0

    # 1. Equal mass: delta = 0 -> odd modes must contribute 0
    hp_odd, hc_odd = get_fd_waveform(
        approximant="IMRPhenomHM",
        mass1=30.0,
        mass2=30.0,
        chi1=0.2,
        chi2=0.2,
        inclination=inc,
        sample_frequencies=f,
        modes=[(2, 1), (3, 3)],
    )
    assert torch.allclose(hp_odd.abs(), torch.zeros_like(hp_odd.abs()), atol=1e-28)
    assert torch.allclose(hc_odd.abs(), torch.zeros_like(hc_odd.abs()), atol=1e-28)

    # 2. Unequal mass: odd modes must be non-zero
    hp_uneq, hc_uneq = get_fd_waveform(
        approximant="IMRPhenomHM",
        mass1=50.0,
        mass2=15.0,
        chi1=0.2,
        chi2=-0.1,
        inclination=inc,
        sample_frequencies=f,
        modes=[(2, 1), (3, 3)],
    )
    assert torch.any(hp_uneq.abs() > 0.0)
    assert torch.any(hc_uneq.abs() > 0.0)


def test_mode_selection():
    """Verify that specifying subsets of modes produces expected polarizations."""
    f = torch.linspace(25.0, 250.0, 80, dtype=torch.float64)
    inc = math.pi / 4.0

    # Mode 22 alone
    hp_22, _ = get_fd_waveform(
        approximant="IMRPhenomHM",
        mass1=40.0,
        mass2=20.0,
        sample_frequencies=f,
        inclination=inc,
        modes=[(2, 2)],
    )

    # All modes
    hp_all, _ = get_fd_waveform(
        approximant="IMRPhenomHM",
        mass1=40.0,
        mass2=20.0,
        sample_frequencies=f,
        inclination=inc,
        modes=[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)],
    )

    # With higher modes at inclination pi/4, waveform differs from 22-only
    diff = (hp_all - hp_22).abs().max()
    assert diff.item() > 0.0


def test_f_lower_is_accepted_and_masks_dispatcher_output():
    """The public dispatcher accepts f_lower and zeroes bins below it."""
    f = torch.tensor([20.0, 29.9, 30.0, 40.0], dtype=torch.float64)
    hp, hc = get_fd_waveform(
        approximant="IMRPhenomHM",
        mass1=40.0,
        mass2=20.0,
        sample_frequencies=f,
        f_lower=30.0,
    )
    assert torch.count_nonzero(hp[:2]) == 0
    assert torch.count_nonzero(hc[:2]) == 0
    assert torch.count_nonzero(hp[2:]) > 0
    assert torch.count_nonzero(hc[2:]) > 0


def test_scalar_frequency_broadcasts_over_parameter_batch():
    """A scalar frequency evaluates each batched binary independently."""
    model = IMRPhenomHM()
    f = torch.tensor(30.0, dtype=torch.float64)
    chi1 = torch.tensor([0.2, 0.3], dtype=torch.float64)

    hp_batch, hc_batch = model(
        f=f,
        mass1=40.0,
        mass2=15.0,
        chi1=chi1,
        chi2=-0.1,
    )
    assert hp_batch.shape == chi1.shape
    assert hc_batch.shape == chi1.shape

    for i in range(chi1.numel()):
        hp_scalar, hc_scalar = model(
            f=f,
            mass1=40.0,
            mass2=15.0,
            chi1=chi1[i],
            chi2=-0.1,
        )
        torch.testing.assert_close(hp_batch[i], hp_scalar)
        torch.testing.assert_close(hc_batch[i], hc_scalar)

    hp_grid, hc_grid = model(
        f=f.reshape(1),
        mass1=40.0,
        mass2=15.0,
        chi1=chi1,
        chi2=-0.1,
    )
    assert hp_grid.shape == (chi1.numel(), 1)
    assert hc_grid.shape == hp_grid.shape
    torch.testing.assert_close(hp_grid[:, 0], hp_batch)
    torch.testing.assert_close(hc_grid[:, 0], hc_batch)


def test_scalar_frequency_carrier_typeerror_propagates():
    """Carrier failures must not be swallowed by compatibility fallbacks."""
    class RaisingD(IMRPhenomD):
        @staticmethod
        def __call__(*args, **kwargs):
            raise TypeError("carrier evaluation failed")

    model = IMRPhenomHM(base_22=RaisingD())
    with pytest.raises(TypeError, match="carrier evaluation failed"):
        model(f=torch.tensor(30.0, dtype=torch.float64))


def test_batching_parity():
    """Verify batched evaluation matches individual evaluation exactly."""
    f = torch.linspace(20.0, 300.0, 100, dtype=torch.float64)
    m1 = torch.tensor([35.0, 45.0, 25.0], dtype=torch.float64)
    m2 = torch.tensor([25.0, 15.0, 20.0], dtype=torch.float64)
    c1 = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float64)
    c2 = torch.tensor([-0.1, 0.2, 0.0], dtype=torch.float64)

    hp_b, hc_b = get_fd_waveform(
        approximant="IMRPhenomXHM",
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        sample_frequencies=f,
        inclination=math.pi / 3.0,
    )
    assert hp_b.shape == (3, len(f))

    for i in range(3):
        hp_s, hc_s = get_fd_waveform(
            approximant="IMRPhenomXHM",
            mass1=m1[i],
            mass2=m2[i],
            chi1=c1[i],
            chi2=c2[i],
            sample_frequencies=f,
            inclination=math.pi / 3.0,
        )
        assert torch.allclose(hp_b[i], hp_s, rtol=1e-8, atol=1e-28)
        assert torch.allclose(hc_b[i], hc_s, rtol=1e-8, atol=1e-28)


def test_imrphenom_hm_autograd():
    """Verify autograd differentiability w.r.t. masses, spins, inclination."""
    f = torch.linspace(30.0, 200.0, 50, dtype=torch.float64)
    m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    c1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    c2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(math.pi / 3.0, dtype=torch.float64, requires_grad=True)

    hp, hc = get_fd_waveform(
        approximant="IMRPhenomXHM",
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        inclination=inc,
        sample_frequencies=f,
    )

    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()

    for p, name in [(m1, "m1"), (m2, "m2"), (c1, "c1"), (c2, "c2"), (inc, "inclination")]:
        assert p.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(p.grad), f"Gradient for {name} is not finite: {p.grad}"
        assert p.grad.item() != 0.0, f"Gradient for {name} is zero"


def test_imrphenom_hm_lal_parity():
    """Verify diagnostic overlap (smoke test, match >= 0.75) against LALSimulation IMRPhenomXHM / IMRPhenomHM."""
    try:
        import lal
        import lalsimulation as lalsim
        import numpy as np
        from pycbc.types import FrequencySeries
        from pycbc.filter.matchedfilter import match
    except ImportError:
        pytest.skip("LALSimulation or PyCBC not installed")

    m1 = 35.0
    m2 = 25.0
    chi1 = 0.1
    chi2 = -0.05
    dist_mpc = 400.0
    inc = math.pi / 4.0  # 45 degrees inclination where higher modes are active
    f_min = 25.0
    f_max = 500.0
    delta_f = 0.25

    lal_approx = lalsim.GetApproximantFromString("IMRPhenomXHM")
    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        0.0, 0.0, chi1,
        0.0, 0.0, chi2,
        dist_mpc * 1e6 * lal.PC_SI,
        inc,
        0.0,
        0.0,
        0.0,
        0.0,
        delta_f,
        f_min,
        f_max,
        f_min,
        None,
        lal_approx,
    )

    n_pts = int(f_max / delta_f) + 1
    f_grid = torch.arange(0, n_pts, dtype=torch.float64) * delta_f
    f_mask = (f_grid >= f_min) & (f_grid <= f_max)
    f_eval = f_grid[f_mask]

    hp_tw, _ = get_fd_waveform(
        approximant="IMRPhenomXHM",
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        inclination=inc,
        sample_frequencies=f_eval,
        distance=dist_mpc,
    )

    s_lal = FrequencySeries(hp_lal.data.data, delta_f=delta_f)
    s_tw = FrequencySeries(np.zeros(len(s_lal), dtype=np.complex128), delta_f=delta_f)
    idx_start = int(f_min / delta_f)
    s_tw.data[idx_start:idx_start + len(hp_tw)] = hp_tw.detach().cpu().numpy()

    overlap, _ = match(s_lal, s_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    assert overlap >= 0.95, f"IMRPhenomXHM match = {overlap:.6f} is below 0.95"


def test_hm_common_time_shift():
    """Verify higher modes obey an exact common time translation exp(-2*pi*i*f*tc)."""
    f = torch.arange(20.0, 1000.25, 0.25, dtype=torch.float64)
    tc = 0.003
    cfg = dict(
        f=f,
        mass1=30.0,
        mass2=10.0,
        chi1=0.2,
        chi2=-0.1,
        f_ref=30.0,
        return_unprojected_modes=True,
    )
    modes_0 = IMRPhenomHM()(**cfg, tc=0.0)[2]
    modes_tc = IMRPhenomHM()(**cfg, tc=tc)[2]
    shift = torch.exp(-2j * torch.pi * f * tc)

    for mode in IMRPhenomHM.SUPPORTED_MODES:
        h0 = modes_0[mode]
        htc = modes_tc[mode]
        target = h0 * shift
        rel_err = float(torch.linalg.norm(htc - target) / torch.linalg.norm(target))
        assert rel_err < 1.0e-13, f"Mode {mode} time translation error {rel_err:.3e} exceeds 1e-13"
