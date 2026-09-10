"""Tests for IMRPhenomP precessing waveform approximant and parity against LALSuite."""

import pytest
import torch
import numpy as np

from torchwave.approximants.imrphenomp import IMRPhenomP
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.waveforms import get_fd_waveform


def test_imrphenomp_aligned_spin_limit():
    """Verify that in the aligned-spin limit (chip=0), IMRPhenomP face-on reduces to IMRPhenomD."""
    f = torch.linspace(20.0, 300.0, 100, dtype=torch.float64)
    m1, m2 = 20.0, 10.0
    s1z, s2z = 0.3, -0.2

    # PhenomD face-on
    hp_d, hc_d = IMRPhenomD()(f, m1, m2, s1z, s2z, distance=100.0, inclination=0.0)

    # PhenomP face-on with zero in-plane spins
    hp_p, hc_p = IMRPhenomP()(
        f,
        mass1=m1,
        mass2=m2,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=s1z,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=s2z,
        distance=100.0,
        inclination=0.0,
    )

    # Should match closely (up to reference phase alignment)
    assert torch.allclose(hp_p.abs(), hp_d.abs(), rtol=1e-5, atol=1e-25)


def test_imrphenomp_batched_evaluation():
    """Verify batching across masses, 3D spins, and inclinations."""
    B = 3
    N_f = 128
    f = torch.linspace(20.0, 300.0, N_f, dtype=torch.float64)
    m1 = torch.tensor([25.0, 30.0, 15.0], dtype=torch.float64)
    m2 = torch.tensor([10.0, 15.0, 10.0], dtype=torch.float64)
    s1x = torch.tensor([0.2, 0.0, 0.3], dtype=torch.float64)
    s1y = torch.tensor([0.1, 0.3, 0.0], dtype=torch.float64)
    s1z = torch.tensor([0.2, -0.1, 0.2], dtype=torch.float64)
    inc = torch.tensor([0.3, 0.6, 0.0], dtype=torch.float64)

    model = IMRPhenomP()
    hp_b, hc_b = model(
        f=f,
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        inclination=inc,
    )

    assert hp_b.shape == (B, N_f)
    assert hc_b.shape == (B, N_f)

    for b in range(B):
        hp_s, hc_s = model(
            f=f,
            mass1=m1[b],
            mass2=m2[b],
            spin1x=s1x[b],
            spin1y=s1y[b],
            spin1z=s1z[b],
            inclination=inc[b],
        )
        assert torch.allclose(hp_b[b], hp_s, atol=1e-25)
        assert torch.allclose(hc_b[b], hc_s, atol=1e-25)


def test_imrphenomp_autograd():
    """Verify full gradient propagation through all parameters."""
    f = torch.linspace(20.0, 200.0, 64, dtype=torch.float64)
    m1 = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
    s1x = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    s1y = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    s1z = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    dist = torch.tensor(150.0, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomP()(
        f,
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        distance=dist,
        inclination=inc,
    )

    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()

    for p_name, p in [
        ("m1", m1),
        ("m2", m2),
        ("s1x", s1x),
        ("s1y", s1y),
        ("s1z", s1z),
        ("dist", dist),
        ("inc", inc),
    ]:
        assert p.grad is not None, f"Gradient for {p_name} is None"
        assert not torch.isnan(p.grad), f"Gradient for {p_name} is NaN"


def test_imrphenomp_lal_parity():
    """Verify match > 0.98 against LALSuite IMRPhenomPv2 for precessing configurations."""
    try:
        import lal
        import lalsimulation as lalsim
        from pycbc.filter.matchedfilter import match
        from pycbc.types import FrequencySeries
    except ImportError:
        pytest.skip("LALSuite or PyCBC not installed in current environment")

    test_cases = [
        # (m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inc, min_match)
        (20.0, 10.0, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.5, 0.98),
        (15.0, 12.0, 0.2, 0.2, -0.1, 0.1, 0.0, 0.2, 0.3, 0.98),
    ]

    df, f_min, f_max, f_ref = 0.25, 20.0, 300.0, 20.0
    for m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, inc, min_match in test_cases:
        hp_lal, hc_lal = lalsim.SimInspiralChooseFDWaveform(
            m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
            s1x, s1y, s1z, s2x, s2y, s2z,
            100.0 * 1e6 * lal.PC_SI, inc, 0.0, 0.0, 0.0, 0.0,
            df, f_min, f_max, f_ref, lal.CreateDict(), lalsim.IMRPhenomPv2
        )
        s_hp_lal = FrequencySeries(hp_lal.data.data, delta_f=df)
        freqs = torch.arange(len(s_hp_lal), dtype=torch.float64) * df

        hp_tw, hc_tw = get_fd_waveform(
            "IMRPhenomP",
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
            sample_frequencies=freqs,
            f_ref=f_ref,
        )

        s_hp_tw = FrequencySeries(hp_tw.detach().cpu().numpy(), delta_f=df)
        overlap, _ = match(s_hp_lal, s_hp_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
        assert overlap >= min_match, f"Match {overlap:.6f} below {min_match} for config m=({m1},{m2}) s1=({s1x},{s1y},{s1z})"
