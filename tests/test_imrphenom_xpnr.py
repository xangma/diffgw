"""Unit and integration tests for the provisional XPNR-labelled route.

Tests the native PNR (PhenomX Precession Numerical Relativity) angle engine,
Ansatz fits, non-precessing collapse to IMRPhenomXHM, batched execution,
autograd differentiability, waveform dispatch aliases, and diagnostic parity
against LALSimulation's IMRPhenomXPNR.  The latter are acceptance gates, not
observed domain bounds or qualification of the published full model.
"""

import pytest
import torch
import numpy as np

from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.imrphenom_hm import IMRPhenomXHM
from torchwave.approximants.precession_pnr import (
    compute_pnr_single_spin,
    compute_pnr_alpha_coefficients,
    compute_pnr_beta_coefficients,
    pnr_angles_window,
    evaluate_pnr_precession,
)
from torchwave import get_fd_waveform


def test_pnr_fits_unit():
    """Verify PNR single-spin mapping and polynomial coefficient evaluations."""
    m1 = torch.tensor(30.0, dtype=torch.float64)
    m2 = torch.tensor(20.0, dtype=torch.float64)
    s1x, s1y, s1z = torch.tensor(0.2), torch.tensor(0.0), torch.tensor(0.3)
    s2x, s2y, s2z = torch.tensor(0.1), torch.tensor(0.0), torch.tensor(-0.2)

    chi_mag, costheta, costheta_f, chi_eff = compute_pnr_single_spin(
        m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )

    assert chi_mag.item() > 0.0
    assert -1.0 <= costheta.item() <= 1.0
    assert -1.0 <= costheta_f.item() <= 1.0

    q = m1 / m2
    eta = (m1 * m2) / ((m1 + m2) ** 2)

    a1, a2, a3, a4 = compute_pnr_alpha_coefficients(eta, chi_mag, costheta, q, m1.device, m1.dtype)
    b0, b1, b2, b3, b4, b5 = compute_pnr_beta_coefficients(eta, chi_mag, costheta, costheta_f, q, m1.device, m1.dtype)

    assert a1.item() >= 0.0
    assert a2.item() <= 0.0
    assert a3.item() > 0.0
    assert a4.item() > 0.0

    assert b4.item() >= 175.0
    assert b5.item() <= 0.0

    # Calibration window
    w = pnr_angles_window(q, chi_mag)
    assert w.item() == 1.0  # (q=1.5, chi ~ 0.3 is well inside calibration)


def test_pnr_angles_evaluation():
    """Verify evaluate_pnr_precession returns finite, physically bounded angles."""
    f = torch.linspace(20.0, 300.0, 150, dtype=torch.float64)
    m1, m2 = 36.0, 24.0
    s1 = torch.tensor([0.2, 0.1, 0.3], dtype=torch.float64)
    s2 = torch.tensor([0.1, -0.1, -0.2], dtype=torch.float64)

    alpha, beta, gamma, theta_jn, zeta, cp = evaluate_pnr_precession(
        f=f,
        f_ref=20.0,
        mass1=m1,
        mass2=m2,
        spin1x=s1[0],
        spin1y=s1[1],
        spin1z=s1[2],
        spin2x=s2[0],
        spin2y=s2[1],
        spin2z=s2[2],
        inclination=0.5,
        phi_ref=0.0,
    )

    assert torch.isfinite(alpha).all()
    assert torch.isfinite(beta).all()
    assert torch.isfinite(gamma).all()
    assert torch.all(beta >= 0.0 - 1e-10) and torch.all(beta <= np.pi + 1e-10)


def test_imrphenom_xpnr_nonprecessing_collapse():
    """Verify that in the non-precessing limit (chi_p -> 0), IMRPhenomXPNR collapses to IMRPhenomXHM."""
    f = torch.linspace(25.0, 300.0, 100, dtype=torch.float64)
    m1, m2 = 35.0, 20.0
    s1z, s2z = 0.3, -0.2
    inc = 0.7
    phi0 = 0.4
    dist = 100.0

    xhm = IMRPhenomXHM()
    xpnr = IMRPhenomXPHM(precession_method="pnr", prec_version=330, final_spin_mod=2)

    hp_xhm, hc_xhm = xhm(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=s1z,
        chi2=s2z,
        distance=dist,
        inclination=inc,
        phi0=phi0,
    )

    hp_pnr, hc_pnr = xpnr(
        f=f,
        mass1=m1,
        mass2=m2,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=s1z,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=s2z,
        distance=dist,
        inclination=inc,
        phi0=phi0,
    )

    diff_p = torch.max(torch.abs(hp_pnr - hp_xhm)).item()
    diff_c = torch.max(torch.abs(hc_pnr - hc_xhm)).item()
    max_amp = torch.max(torch.abs(hp_xhm)).item()

    rel_diff_p = diff_p / max_amp
    rel_diff_c = diff_c / max_amp

    assert rel_diff_p < 1e-12, f"Relative difference in h_plus {rel_diff_p} exceeds tolerance"
    assert rel_diff_c < 1e-12, f"Relative difference in h_cross {rel_diff_c} exceeds tolerance"


def test_imrphenom_xpnr_batching():
    """Verify batched evaluation across masses and 3D spins matches sequential calls."""
    B = 3
    N_f = 64
    f = torch.linspace(25.0, 250.0, N_f, dtype=torch.float64)

    m1_b = torch.tensor([30.0, 40.0, 35.0], dtype=torch.float64)
    m2_b = torch.tensor([20.0, 25.0, 20.0], dtype=torch.float64)
    s1x_b = torch.tensor([0.2, 0.1, 0.2], dtype=torch.float64)
    s1y_b = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64)
    s1z_b = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    s2x_b = torch.tensor([0.0, 0.1, 0.1], dtype=torch.float64)
    s2y_b = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64)
    s2z_b = torch.tensor([-0.1, 0.0, -0.2], dtype=torch.float64)
    inc_b = torch.tensor([0.4, 0.7, 0.5], dtype=torch.float64)

    model = IMRPhenomXPHM(modes=[(2, 2)], precession_method="pnr", prec_version=330, final_spin_mod=2)
    hp_b, hc_b = model(
        f=f,
        mass1=m1_b,
        mass2=m2_b,
        spin1x=s1x_b,
        spin1y=s1y_b,
        spin1z=s1z_b,
        spin2x=s2x_b,
        spin2y=s2y_b,
        spin2z=s2z_b,
        inclination=inc_b,
    )

    assert hp_b.shape == (B, N_f)
    assert hc_b.shape == (B, N_f)

    for b in range(B):
        hp_s, hc_s = model(
            f=f,
            mass1=m1_b[b],
            mass2=m2_b[b],
            spin1x=s1x_b[b],
            spin1y=s1y_b[b],
            spin1z=s1z_b[b],
            spin2x=s2x_b[b],
            spin2y=s2y_b[b],
            spin2z=s2z_b[b],
            inclination=inc_b[b],
        )
        assert torch.allclose(hp_b[b], hp_s, atol=1e-25, rtol=1e-5)
        assert torch.allclose(hc_b[b], hc_s, atol=1e-25, rtol=1e-5)


def test_imrphenom_xpnr_dispatch():
    """Verify get_fd_waveform high-level dispatch with alias names."""
    f = torch.linspace(25.0, 200.0, 50, dtype=torch.float64)
    m1, m2 = 35.0, 20.0
    s1x, s1y, s1z = 0.2, 0.1, 0.1
    s2x, s2y, s2z = 0.0, 0.0, -0.1
    inc = 0.5

    hp_pnr, hc_pnr = get_fd_waveform(
        "IMRPhenomXPNR",
        mass1=m1, mass2=m2,
        spin1x=s1x, spin1y=s1y, spin1z=s1z,
        spin2x=s2x, spin2y=s2y, spin2z=s2z,
        inclination=inc,
        sample_frequencies=f,
    )

    hp_npr, hc_npr = get_fd_waveform(
        "IMRPhenomXNPR",
        mass1=m1, mass2=m2,
        spin1x=s1x, spin1y=s1y, spin1z=s1z,
        spin2x=s2x, spin2y=s2y, spin2z=s2z,
        inclination=inc,
        sample_frequencies=f,
    )

    assert torch.allclose(hp_pnr, hp_npr, atol=1e-25)
    assert torch.allclose(hc_pnr, hc_npr, atol=1e-25)


def test_imrphenom_xpnr_autograd():
    """Verify PyTorch autograd gradients through IMRPhenomXPNR."""
    f = torch.linspace(25.0, 200.0, 32, dtype=torch.float64)

    m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    s1x = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    s1y = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    s1z = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomXPHM(modes=[(2, 2)], precession_method="pnr", prec_version=330, final_spin_mod=2)(
        f=f,
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        inclination=inc,
    )

    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()

    for name, param in [("m1", m1), ("m2", m2), ("s1x", s1x), ("s1y", s1y), ("s1z", s1z), ("inc", inc)]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad), f"Gradient for {name} is NaN"
        assert param.grad.abs().item() > 0.0, f"Gradient for {name} is zero"


@pytest.mark.parametrize(
    "m1,m2,s1x,s1y,s1z,s2x,s2y,s2z,inc,min_match",
    [
        (30.0, 20.0, 0.2, 0.0, 0.3, 0.1, 0.0, -0.2, 0.5, 0.99),
        (35.0, 25.0, 0.1, 0.2, -0.1, 0.0, 0.1, 0.3, 0.8, 0.99),
        (40.0, 30.0, 0.2, 0.0, 0.3, 0.1, 0.0, -0.2, 0.5, 0.99),
        (50.0, 30.0, 0.2, -0.1, 0.4, 0.2, 0.1, -0.3, 0.3, 0.99),
    ],
    ids=("q1p5", "q1p4", "q1p33", "q1p67"),
)
def test_imrphenom_xpnr_lal_parity(
    request, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inc, min_match
):
    """Apply the declared black-box match gate to one configuration.

    Parametrization keeps every failed configuration visible.  The threshold
    is not an observed domain bound or a full-model qualification.
    """
    try:
        import lal
        import lalsimulation as lalsim
        from pycbc.filter.matchedfilter import match
        from pycbc.types import FrequencySeries
    except ImportError:
        pytest.skip("LALSimulation or PyCBC not installed in environment")

    df, f_min, f_max, f_ref = 0.25, 20.0, 400.0, 20.0
    p = lal.CreateDict()
    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        s1x, s1y, s1z,
        s2x, s2y, s2z,
        100.0 * 1e6 * lal.PC_SI,
        inc, 0.0, 0.0, 0.0, 0.0,
        df, f_min, f_max, f_ref,
        p,
        lalsim.IMRPhenomXPNR,
    )
    s_hp_lal = FrequencySeries(hp_lal.data.data, delta_f=df)
    freqs = torch.arange(len(s_hp_lal), dtype=torch.float64) * df

    hp_tw, _ = get_fd_waveform(
        "IMRPhenomXPNR",
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=s2x,
        spin2y=s2y,
        spin2z=s2z,
        distance=100.0,
        inclination=inc,
        f_ref=f_ref,
        sample_frequencies=freqs,
    )

    s_hp_tw = FrequencySeries(hp_tw.detach().cpu().numpy(), delta_f=df)
    overlap, _ = match(s_hp_lal, s_hp_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    request.node.user_properties.extend(
        (("observed_match", f"{overlap:.12f}"), ("acceptance_threshold", f"{min_match:.2f}"))
    )
    assert overlap >= min_match, (
        f"Match {overlap:.8f} below {min_match} for m=({m1},{m2}) "
        f"s1=({s1x},{s1y},{s1z})"
    )
