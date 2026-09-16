"""Unit and integration tests for IMRPhenomXPHM-SpinTaylor approximant.

Tests sovereign SpinTaylorT4 numerical precession engine, angle evaluation,
non-precessing collapse to IMRPhenomXHM, batched execution, autograd differentiability,
waveform dispatch aliases, and diagnostic parity against LALSimulation.
"""

import pytest
import torch
import numpy as np

from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.imrphenom_hm import IMRPhenomXHM
from torchwave.approximants.precession_spintaylor import (
    SpinTaylorT4Engine,
    evaluate_spintaylor_precession,
    compute_chi_tot_perp,
)
from torchwave import get_fd_waveform


def test_spintaylor_engine_unit():
    """Verify SpinTaylorT4Engine conservation laws and ODE stepping."""
    m1, m2 = 35.0, 20.0
    s1 = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    s2 = torch.tensor([0.1, -0.2, 0.2], dtype=torch.float64)
    engine = SpinTaylorT4Engine(m1, m2, s1, s2, f_ref=20.0)

    # Initial velocities and spin norms
    v0 = engine.v_ref.item()
    assert 0.0 < v0 < 1.0

    s1_norm_0 = torch.norm(s1).item()
    s2_norm_0 = torch.norm(s2).item()

    # Step ODE from v0 to v_max
    v_nodes, L_nodes, S1_nodes, S2_nodes = engine.integrate_nodes(
        v_min=v0 * 0.8,
        v_max=0.35,
        num_steps=150,
    )

    # Monotonicity of velocity
    assert torch.all(v_nodes[1:] >= v_nodes[:-1])

    # Conservation of spin magnitudes along integration trajectory
    s1_norms = torch.norm(S1_nodes, dim=-1)
    s2_norms = torch.norm(S2_nodes, dim=-1)

    assert torch.allclose(s1_norms, torch.full_like(s1_norms, s1_norm_0), atol=1e-10)
    assert torch.allclose(s2_norms, torch.full_like(s2_norms, s2_norm_0), atol=1e-10)


def test_spintaylor_angles_evaluation():
    """Verify evaluate_spintaylor_precession returns finite, physically bounded angles."""
    f = torch.linspace(20.0, 300.0, 150, dtype=torch.float64)
    m1, m2 = 36.0, 29.0
    s1 = torch.tensor([0.3, 0.2, -0.1], dtype=torch.float64)
    s2 = torch.tensor([0.1, -0.2, 0.2], dtype=torch.float64)

    alpha, beta, gamma, theta_jn, zeta, carrier_phase = evaluate_spintaylor_precession(
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
        inclination=0.6,
        phi_ref=0.0,
    )

    assert torch.isfinite(alpha).all()
    assert torch.isfinite(beta).all()
    assert torch.isfinite(gamma).all()
    assert torch.all(beta >= 0.0 - 1e-10) and torch.all(beta <= np.pi + 1e-10)


def test_imrphenom_xphm_spintaylor_nonprecessing_collapse():
    """Verify that in the non-precessing limit (chi_p -> 0), SpinTaylor collapses to IMRPhenomXHM."""
    f = torch.linspace(25.0, 300.0, 100, dtype=torch.float64)
    m1, m2 = 35.0, 20.0
    s1z, s2z = 0.3, -0.2
    inc = 0.7
    phi0 = 0.4
    dist = 100.0

    xhm = IMRPhenomXHM()
    xphm_st = IMRPhenomXPHM(precession_method="spintaylor", prec_version=320, final_spin_mod=2)

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

    hp_st, hc_st = xphm_st(
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

    diff_p = torch.max(torch.abs(hp_st - hp_xhm)).item()
    diff_c = torch.max(torch.abs(hc_st - hc_xhm)).item()
    max_amp = torch.max(torch.abs(hp_xhm)).item()

    rel_diff_p = diff_p / max_amp
    rel_diff_c = diff_c / max_amp

    assert rel_diff_p < 1e-12, f"Relative difference in h_plus {rel_diff_p} exceeds tolerance"
    assert rel_diff_c < 1e-12, f"Relative difference in h_cross {rel_diff_c} exceeds tolerance"


def test_imrphenom_xphm_spintaylor_batching():
    """Verify batched evaluation across masses and 3D spins matches sequential calls."""
    B = 3
    N_f = 64
    f = torch.linspace(25.0, 250.0, N_f, dtype=torch.float64)

    m1_b = torch.tensor([30.0, 40.0, 25.0], dtype=torch.float64)
    m2_b = torch.tensor([20.0, 25.0, 15.0], dtype=torch.float64)
    s1x_b = torch.tensor([0.2, 0.1, 0.3], dtype=torch.float64)
    s1y_b = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64)
    s1z_b = torch.tensor([0.2, -0.1, 0.1], dtype=torch.float64)
    s2x_b = torch.tensor([0.0, 0.1, 0.0], dtype=torch.float64)
    s2y_b = torch.tensor([0.0, 0.0, 0.1], dtype=torch.float64)
    s2z_b = torch.tensor([-0.1, 0.0, 0.2], dtype=torch.float64)
    inc_b = torch.tensor([0.4, 0.7, 0.2], dtype=torch.float64)

    model = IMRPhenomXPHM(precession_method="spintaylor", prec_version=320, final_spin_mod=2)
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
        assert torch.allclose(hp_b[b], hp_s, atol=1e-25, rtol=1e-6)
        assert torch.allclose(hc_b[b], hc_s, atol=1e-25, rtol=1e-6)


def test_imrphenom_xphm_spintaylor_dispatch():
    """Verify get_fd_waveform high-level dispatch with alias names and automatic defaults."""
    f = torch.linspace(25.0, 200.0, 50, dtype=torch.float64)
    m1, m2 = 35.0, 20.0
    s1x, s1y, s1z = 0.2, 0.1, 0.1
    s2x, s2y, s2z = 0.0, 0.0, -0.1
    inc = 0.5

    # Direct dispatch via alias
    hp_alias, hc_alias = get_fd_waveform(
        "IMRPhenomXPHM-SpinTaylor",
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=s2x,
        spin2y=s2y,
        spin2z=s2z,
        inclination=inc,
        sample_frequencies=f,
    )

    # Explicit kwargs
    hp_exp, hc_exp = get_fd_waveform(
        "IMRPhenomXPHM",
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=s2x,
        spin2y=s2y,
        spin2z=s2z,
        inclination=inc,
        sample_frequencies=f,
        precession_method="spintaylor",
        prec_version=320,
        final_spin_mod=2,
    )

    assert torch.allclose(hp_alias, hp_exp, atol=1e-25)
    assert torch.allclose(hc_alias, hc_exp, atol=1e-25)


def test_imrphenom_xphm_spintaylor_autograd():
    """Verify PyTorch autograd gradients through IMRPhenomXPHM-SpinTaylor."""
    f = torch.linspace(25.0, 200.0, 32, dtype=torch.float64)

    m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    s1x = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    s1y = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    s1z = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomXPHM(precession_method="spintaylor", prec_version=320, final_spin_mod=2)(
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


def test_imrphenom_xphm_spintaylor_lal_parity():
    """Verify match >= min_match against LALSimulation IMRPhenomXPHM with prec_version=320, final_spin_mod=2."""
    try:
        import lal
        import lalsimulation as lalsim
        from pycbc.filter.matchedfilter import match
        from pycbc.types import FrequencySeries
    except ImportError:
        pytest.skip("LALSimulation or PyCBC not installed in environment")

    test_cases = [
        # (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inc, min_match)
        (35.0, 25.0, 0.2, 0.1, 0.1, 0.0, 0.0, -0.1, 0.6, 0.98),
        (40.0, 30.0, 0.15, 0.1, 0.2, 0.0, 0.0, 0.1, 0.5, 0.98),
        (30.0, 20.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.4, 0.98),
    ]

    df, f_min, f_max, f_ref = 0.25, 20.0, 300.0, 20.0
    for m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inc, min_match in test_cases:
        d_params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d_params, 320)
        lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(d_params, 2)
        hp_lal, hc_lal = lalsim.SimInspiralChooseFDWaveform(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            s1x, s1y, s1z,
            s2x, s2y, s2z,
            100.0 * 1e6 * lal.PC_SI,
            inc, 0.0, 0.0, 0.0, 0.0,
            df, f_min, f_max, f_ref,
            d_params,
            lalsim.IMRPhenomXPHM,
        )
        s_hp_lal = FrequencySeries(hp_lal.data.data, delta_f=df)
        freqs = torch.arange(len(s_hp_lal), dtype=torch.float64) * df

        hp_tw, hc_tw = get_fd_waveform(
            "IMRPhenomXPHM-SpinTaylor",
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
        assert overlap >= min_match, f"Match {overlap:.5f} below {min_match} for m=({m1},{m2}) s1=({s1x},{s1y},{s1z})"
