"""Unit tests for IMRPhenomXPHM precessing multipolar higher-harmonic approximant."""

import pytest
import torch

from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.imrphenom_hm import IMRPhenomXHM
from torchwave import get_fd_waveform


def test_imrphenom_xphm_nonprecessing_collapse():
    """Verify that in the non-precessing limit (chip -> 0), IMRPhenomXPHM collapses to IMRPhenomXHM to machine precision."""
    f = torch.linspace(25.0, 300.0, 100, dtype=torch.float64)
    m1, m2 = 35.0, 20.0
    s1z, s2z = 0.3, -0.2
    inc = 0.7
    phi0 = 0.4
    dist = 100.0

    xhm = IMRPhenomXHM()
    xphm = IMRPhenomXPHM()

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

    hp_xphm, hc_xphm = xphm(
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

    diff_p = torch.max(torch.abs(hp_xphm - hp_xhm)).item()
    diff_c = torch.max(torch.abs(hc_xphm - hc_xhm)).item()
    max_amp = torch.max(torch.abs(hp_xhm)).item()

    rel_diff_p = diff_p / max_amp
    rel_diff_c = diff_c / max_amp

    assert rel_diff_p < 1e-12, f"Relative difference in h_plus {rel_diff_p} exceeds tolerance"
    assert rel_diff_c < 1e-12, f"Relative difference in h_cross {rel_diff_c} exceeds tolerance"


def test_imrphenom_xphm_batching():
    """Verify batched evaluation across masses, 3D spins, and inclinations."""
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

    model = IMRPhenomXPHM()
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
        assert torch.allclose(hp_b[b], hp_s, atol=1e-25)
        assert torch.allclose(hc_b[b], hc_s, atol=1e-25)


def test_imrphenom_xphm_autograd():
    """Verify finite non-zero autograd gradients with respect to parameters."""
    f = torch.linspace(25.0, 200.0, 32, dtype=torch.float64)

    m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    s1x = torch.tensor(0.20, dtype=torch.float64, requires_grad=True)
    s1y = torch.tensor(0.10, dtype=torch.float64, requires_grad=True)
    s1z = torch.tensor(0.20, dtype=torch.float64, requires_grad=True)
    s2x = torch.tensor(-0.10, dtype=torch.float64, requires_grad=True)
    s2y = torch.tensor(0.15, dtype=torch.float64, requires_grad=True)
    s2z = torch.tensor(-0.10, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomXPHM()(
        f=f,
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=s2x,
        spin2y=s2y,
        spin2z=s2z,
        inclination=inc,
        f_ref=20.0,
    )

    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()

    for name, param in [
        ("m1", m1), ("m2", m2), ("s1x", s1x), ("s1y", s1y),
        ("s1z", s1z), ("s2x", s2x), ("s2y", s2y), ("s2z", s2z),
        ("inc", inc),
    ]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad), f"Gradient for {name} is NaN"
        assert param.grad.abs().item() > 0.0, f"Gradient for {name} is zero"


def test_imrphenom_xphm_lal_parity():
    """Verify diagnostic overlap (smoke test, match >= min_match) against LALSimulation IMRPhenomXPHM for precessing configurations."""
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
        lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d_params, 102)
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
            "IMRPhenomXPHM",
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
            precession_method="nnlo",
            prec_version=102,
        )

        s_hp_tw = FrequencySeries(hp_tw.detach().cpu().numpy(), delta_f=df)
        overlap, _ = match(s_hp_lal, s_hp_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
        assert overlap >= min_match, f"Match {overlap:.5f} below {min_match} for m=({m1},{m2}) s1=({s1x},{s1y},{s1z})"
