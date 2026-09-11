"""Tests for clean-room IMRPhenomD_NRTidal and IMRPhenomD_NRTidalv2 approximants.

Literature citations:
- Dietrich et al. (2017), Phys. Rev. D 96, 121501(R) (arXiv:1706.01111)
- Dietrich et al. (2019), Phys. Rev. D 100, 044003 (arXiv:1905.06011)
- Yagi & Yunes (2017), Phys. Rept. 681, 1 (arXiv:1608.02582)
"""

import math
import pytest
import torch

from torchwave.waveforms import get_fd_waveform
from torchwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal,
    IMRPhenomD_NRTidalv2,
    _planck_taper_window,
    _bns_merger_frequency,
    _quadrupole_monopole_constants,
)


def test_universal_i_love_q_relations():
    """Verify universal relations for CQ and COc match Yagi & Yunes (2017) / Dietrich (2019)."""
    lam_zero = torch.tensor(0.0, dtype=torch.float64)
    cq0, co0 = _quadrupole_monopole_constants(lam_zero)
    assert torch.isclose(cq0, torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(co0, torch.tensor(1.0, dtype=torch.float64))

    # For typical NS lambda ~ 400
    lam_ns = torch.tensor(400.0, dtype=torch.float64)
    cq, co = _quadrupole_monopole_constants(lam_ns)
    assert cq.item() > 1.0 and cq.item() < 10.0
    assert co.item() > 1.0 and co.item() < 15.0


def test_bns_merger_frequency():
    """Verify BNS merger frequency fit produces reasonable physical values."""
    m1, m2 = torch.tensor(1.4, dtype=torch.float64), torch.tensor(1.4, dtype=torch.float64)
    lam1, lam2 = torch.tensor(400.0, dtype=torch.float64), torch.tensor(400.0, dtype=torch.float64)
    f_merg = _bns_merger_frequency(m1, m2, lam1, lam2)
    # BNS merger frequency is typically around 1000 - 2000 Hz
    assert 1000.0 < f_merg.item() < 2500.0

    # Stiffer EOS (larger lambda) -> larger radius -> lower merger frequency
    lam_stiff = torch.tensor(1000.0, dtype=torch.float64)
    f_merg_stiff = _bns_merger_frequency(m1, m2, lam_stiff, lam_stiff)
    assert f_merg_stiff < f_merg


def test_planck_taper_properties():
    """Verify Planck taper properties: 1 below f1, 0 above f2, smooth in between."""
    f = torch.linspace(1000.0, 2000.0, 500, dtype=torch.float64)
    f1, f2 = 1200.0, 1600.0
    taper = _planck_taper_window(f, f1, f2)

    assert torch.all(taper[f <= f1] == 1.0)
    assert torch.all(taper[f >= f2] == 0.0)
    assert torch.all((taper >= 0.0) & (taper <= 1.0))
    # Monotonically decreasing in window
    window_mask = (f > f1) & (f < f2)
    window_vals = taper[window_mask]
    diffs = torch.diff(window_vals)
    assert torch.all(diffs <= 0.0)


def test_nrtidal_waveform_generation():
    """Verify basic waveform generation for NRTidal and NRTidalv2."""
    f = torch.linspace(20.0, 2048.0, 1000, dtype=torch.float64)
    for approx in ["IMRPhenomD_NRTidal", "IMRPhenomD_NRTidalv2"]:
        hp, hc = get_fd_waveform(
            approximant=approx,
            mass1=1.4,
            mass2=1.35,
            chi1=0.02,
            chi2=-0.01,
            lambda1=350.0,
            lambda2=420.0,
            sample_frequencies=f,
            distance=40.0,
        )
        assert hp.shape == f.shape
        assert hc.shape == f.shape
        assert torch.all(torch.isfinite(hp))
        assert torch.all(torch.isfinite(hc))
        assert torch.any(hp.abs() > 0.0)


def test_nrtidal_batching_parity():
    """Verify batched evaluation matches individual evaluation exactly."""
    f = torch.linspace(20.0, 1500.0, 300, dtype=torch.float64)
    m1 = torch.tensor([1.4, 1.5, 1.35], dtype=torch.float64)
    m2 = torch.tensor([1.35, 1.25, 1.3], dtype=torch.float64)
    c1 = torch.tensor([0.02, -0.01, 0.0], dtype=torch.float64)
    c2 = torch.tensor([-0.01, 0.03, 0.01], dtype=torch.float64)
    l1 = torch.tensor([350.0, 280.0, 450.0], dtype=torch.float64)
    l2 = torch.tensor([420.0, 390.0, 500.0], dtype=torch.float64)

    hp_b, hc_b = get_fd_waveform(
        approximant="IMRPhenomD_NRTidalv2",
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        lambda1=l1,
        lambda2=l2,
        sample_frequencies=f,
        distance=40.0,
    )
    assert hp_b.shape == (3, len(f))

    for i in range(3):
        hp_s, hc_s = get_fd_waveform(
            approximant="IMRPhenomD_NRTidalv2",
            mass1=m1[i],
            mass2=m2[i],
            chi1=c1[i],
            chi2=c2[i],
            lambda1=l1[i],
            lambda2=l2[i],
            sample_frequencies=f,
            distance=40.0,
        )
        assert torch.allclose(hp_b[i], hp_s, rtol=1e-10, atol=1e-30)
        assert torch.allclose(hc_b[i], hc_s, rtol=1e-10, atol=1e-30)


def test_nrtidal_autograd_gradients():
    """Verify autograd produces finite non-zero gradients w.r.t masses, spins, and tides."""
    f = torch.linspace(30.0, 800.0, 100, dtype=torch.float64)
    m1 = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    c1 = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
    c2 = torch.tensor(-0.01, dtype=torch.float64, requires_grad=True)
    l1 = torch.tensor(350.0, dtype=torch.float64, requires_grad=True)
    l2 = torch.tensor(420.0, dtype=torch.float64, requires_grad=True)

    hp, hc = get_fd_waveform(
        approximant="IMRPhenomD_NRTidalv2",
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        lambda1=l1,
        lambda2=l2,
        sample_frequencies=f,
        distance=50.0,
    )

    loss = (hp.abs() ** 2).sum()
    loss.backward()

    for p, name in [(m1, "m1"), (m2, "m2"), (c1, "c1"), (c2, "c2"), (l1, "lambda1"), (l2, "lambda2")]:
        assert p.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(p.grad), f"Gradient for {name} is not finite: {p.grad}"
        assert p.grad.item() != 0.0, f"Gradient for {name} is zero"


def test_nrtidalv2_lal_parity():
    """Verify match > 0.999 against LALSimulation IMRPhenomD_NRTidalv2."""
    try:
        import lal
        import lalsimulation as lalsim
        import numpy as np
        from pycbc.types import FrequencySeries
        from pycbc.filter.matchedfilter import match
    except ImportError:
        pytest.skip("LALSimulation or PyCBC not installed")

    m1 = 1.4
    m2 = 1.3
    chi1 = 0.02
    chi2 = -0.01
    lambda1 = 350.0
    lambda2 = 420.0
    dist_mpc = 40.0
    f_min = 25.0
    f_max = 1200.0
    delta_f = 0.25

    # LAL generation
    lal_approx = lalsim.GetApproximantFromString("IMRPhenomD_NRTidalv2")
    lal_params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(lal_params, float(lambda1))
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(lal_params, float(lambda2))

    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        0.0, 0.0, chi1,
        0.0, 0.0, chi2,
        dist_mpc * 1e6 * lal.PC_SI,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        delta_f,
        f_min,
        f_max,
        f_min,
        lal_params,
        lal_approx,
    )

    # TorchWave generation
    n_pts = int(f_max / delta_f) + 1
    f_grid = torch.arange(0, n_pts, dtype=torch.float64) * delta_f
    f_mask = (f_grid >= f_min) & (f_grid <= f_max)
    f_eval = f_grid[f_mask]

    hp_tw, _ = get_fd_waveform(
        approximant="IMRPhenomD_NRTidalv2",
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        lambda1=lambda1,
        lambda2=lambda2,
        sample_frequencies=f_eval,
        distance=dist_mpc,
    )

    s_lal = FrequencySeries(hp_lal.data.data, delta_f=delta_f)
    s_tw = FrequencySeries(np.zeros(len(s_lal), dtype=np.complex128), delta_f=delta_f)
    idx_start = int(f_min / delta_f)
    s_tw.data[idx_start:idx_start + len(hp_tw)] = hp_tw.detach().cpu().numpy()

    overlap, _ = match(s_lal, s_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    assert overlap > 0.999, f"IMRPhenomD_NRTidalv2 match = {overlap:.6f} is below 0.999"
