"""Tests for clean-room LISA Suite (TDI, noise PSDs, constellation geometry, LISANetwork).

Literature citations:
- Babak et al. (2021), Phys. Rev. D 103, 084014 (arXiv:2101.05872)
- Robson, Cornish & Liu (2019), Class. Quantum Grav. 36, 105011 (arXiv:1803.01944)
- Tinto & Dhurandhar (2005), Living Rev. Relativ. 8, 4 (arXiv:gr-qc/0409034)
- Vallisneri (2005), Phys. Rev. D 71, 022001 (arXiv:gr-qc/0407102)
"""

import math
import pytest
import torch

from torchwave.detectors.lisa import (
    LISANetwork,
    lisa_psd,
    lisa_oms_displacement_noise,
    lisa_test_mass_acceleration_noise,
    lisa_galactic_foreground_noise,
    LISA_ARM_LENGTH,
    LISA_TRANSFER_FREQ,
)
from torchwave.waveforms import get_fd_waveform
from torchwave.inference.fisher import optimal_snr


def test_lisa_constants_and_geometry():
    """Verify LISA constellation fundamental scale and transfer frequency."""
    assert math.isclose(LISA_ARM_LENGTH, 2.5e9, rel_tol=1e-5)
    expected_f_star = 299792458.0 / (2.0 * math.pi * 2.5e9)
    assert math.isclose(LISA_TRANSFER_FREQ, expected_f_star, rel_tol=1e-5)
    assert math.isclose(LISA_TRANSFER_FREQ * 1000.0, 19.08538, abs_tol=0.01)


def test_tdi_tensors_orthogonality_and_null_channel():
    """Verify TDI A and E response tensors are orthogonal, equal trace, and T is null."""
    tensors = LISANetwork.detector_tensors()
    da = tensors["A"]
    de = tensors["E"]
    dt = tensors["T"]

    # 1. Closed triangle implies T tensor identically vanishes
    assert torch.allclose(dt, torch.zeros_like(dt), atol=1e-12)

    # 2. Orthogonality: Da : De = 0
    inner_ae = torch.sum(da * de).item()
    assert abs(inner_ae) < 1e-12, f"Da : De = {inner_ae} is not zero"

    # 3. Symmetry and equal variance: Tr(Da^2) == Tr(De^2) == 3/8
    tr_da2 = torch.sum(da * da).item()
    tr_de2 = torch.sum(de * de).item()
    assert math.isclose(tr_da2, 0.375, rel_tol=1e-5)
    assert math.isclose(tr_de2, 0.375, rel_tol=1e-5)


def test_lisa_psd_symmetry_and_sagnac_suppression():
    """Verify S_A(f) == S_E(f) and S_T(f) << S_A(f) at low frequencies."""
    f = torch.logspace(-4, 0, 100, dtype=torch.float64)  # 0.1 mHz to 1 Hz

    s_a = lisa_psd(f, channel="A", include_galactic_confusion=False)
    s_e = lisa_psd(f, channel="E", include_galactic_confusion=False)
    s_t = lisa_psd(f, channel="T", include_galactic_confusion=False)

    # 1. Exact equality S_A(f) == S_E(f)
    assert torch.allclose(s_a, s_e, rtol=1e-12, atol=1e-45)

    # 2. Positive definite PSD
    assert torch.all(s_a > 0.0)
    assert torch.all(s_t > 0.0)

    # 3. At 1 mHz (f << f_*), T channel must be suppressed by at least 10^(-3)
    idx_1mhz = torch.argmin(torch.abs(f - 1.0e-3))
    suppression_1mhz = (s_t[idx_1mhz] / s_a[idx_1mhz]).item()
    assert suppression_1mhz < 1.0e-3, f"Suppression {suppression_1mhz} at 1 mHz is not < 1e-3"

    # 4. At 0.1 mHz, suppression must be even stronger
    idx_01mhz = torch.argmin(torch.abs(f - 1.0e-4))
    suppression_01mhz = (s_t[idx_01mhz] / s_a[idx_01mhz]).item()
    assert suppression_01mhz < 1.0e-6


def test_lisa_antenna_patterns_autograd():
    """Verify antenna patterns for A, E, T and PyTorch autograd gradients."""
    net = LISANetwork()

    th = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    ph = torch.tensor(2.4, dtype=torch.float64, requires_grad=True)
    ps = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)

    fp_a, fc_a = net.antenna_patterns(th, ph, ps, channel="A")
    fp_e, fc_e = net.antenna_patterns(th, ph, ps, channel="E")
    fp_t, fc_t = net.antenna_patterns(th, ph, ps, channel="T")

    # T channel antenna response is identically zero in long-wavelength limit
    assert abs(fp_t.item()) < 1e-12
    assert abs(fc_t.item()) < 1e-12

    # A and E patterns are non-zero
    assert abs(fp_a.item()) > 0.0 or abs(fc_a.item()) > 0.0
    assert abs(fp_e.item()) > 0.0 or abs(fc_e.item()) > 0.0

    # Test autograd backward
    loss = fp_a ** 2 + fc_a ** 2 + fp_e ** 2 + fc_e ** 2
    loss.backward()

    assert th.grad is not None and torch.isfinite(th.grad)
    assert ph.grad is not None and torch.isfinite(ph.grad)


def test_lisa_smbhb_and_emri_evaluation():
    """Evaluate supermassive black hole binary in LISA mHz band and compute optimal SNR."""
    net = LISANetwork()

    # SMBHB: 2 x 10^6 M_sun at z ~ 1 (D_L ~ 6.7 Gpc)
    m1 = 1.2e6
    m2 = 0.8e6
    f_grid = torch.linspace(1.0e-4, 5.0e-2, 200, dtype=torch.float64)  # 0.1 mHz to 50 mHz

    hp, hc = get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        distance=6700.0,
        sample_frequencies=f_grid,
    )

    theta = math.pi / 3.0
    phi = math.pi / 4.0
    psi = 0.2

    # Project into A and E TDI channels
    h_a = net.project_fd_strain(hp, hc, f_grid, theta, phi, psi, channel="A")
    h_e = net.project_fd_strain(hp, hc, f_grid, theta, phi, psi, channel="E")

    assert h_a.shape == (200,)
    assert h_e.shape == (200,)
    assert torch.all(torch.isfinite(h_a))
    assert torch.all(torch.isfinite(h_e))

    # Compute optimal SNR in A and E channels
    psd_a = net.psd(f_grid, channel="A")
    psd_e = net.psd(f_grid, channel="E")

    snr_a = optimal_snr(waveform=h_a, psd=psd_a, frequencies=f_grid).item()
    snr_e = optimal_snr(waveform=h_e, psd=psd_e, frequencies=f_grid).item()
    total_snr = math.sqrt(snr_a ** 2 + snr_e ** 2)

    assert total_snr > 10.0, f"Expected loud SMBHB detection (SNR > 10), got {total_snr:.2f}"
