"""Tests for TaylorF2 higher-order spin corrections and parity against LALSuite."""

import pytest
import torch

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.waveforms import get_fd_waveform


def test_taylorf2_spin_order_effects():
    """Verify that spin_order affects the higher-order phase coefficients as expected."""
    m1 = torch.tensor(1.4, dtype=torch.float64)
    m2 = torch.tensor(1.4, dtype=torch.float64)
    chi1 = torch.tensor(0.4, dtype=torch.float64)
    chi2 = torch.tensor(0.3, dtype=torch.float64)

    coeffs_so4 = TaylorF2.phase_coefficients(m1, m2, chi1, chi2, spin_order=4)
    coeffs_so7 = TaylorF2.phase_coefficients(m1, m2, chi1, chi2, spin_order=7)

    # 0PN, 1PN, 1.5PN, 2PN should match
    for i in range(4):
        assert torch.allclose(coeffs_so4[i], coeffs_so7[i])

    # 3PN (index 6) differs due to both spin-orbit and spin-spin additions.
    assert not torch.allclose(coeffs_so4[6], coeffs_so7[6])

    # 3.5PN (index 8) differs due to spin-orbit; cubic spin requires order >= 8.
    assert not torch.allclose(coeffs_so4[8], coeffs_so7[8])


def test_taylorf2_autograd_gradients():
    """Verify differentiability with respect to masses and spins."""
    f = torch.linspace(30.0, 500.0, 100, dtype=torch.float64)
    m1 = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    hp, hc = TaylorF2()(f, m1, m2, chi1, chi2)
    loss = (hp.abs() ** 2).sum()
    loss.backward()

    assert m1.grad is not None and not torch.isnan(m1.grad)
    assert m2.grad is not None and not torch.isnan(m2.grad)
    assert chi1.grad is not None and not torch.isnan(chi1.grad)
    assert chi2.grad is not None and not torch.isnan(chi2.grad)


def test_taylorf2_batched_generation():
    """Verify batched waveform generation matches individual waveforms."""
    B = 3
    N_f = 200
    f = torch.linspace(30.0, 600.0, N_f, dtype=torch.float64)
    m1 = torch.tensor([1.4, 1.6, 2.0], dtype=torch.float64)
    m2 = torch.tensor([1.4, 1.2, 1.5], dtype=torch.float64)
    chi1 = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    chi2 = torch.tensor([0.1, 0.3, -0.2], dtype=torch.float64)

    hp_batch, hc_batch = TaylorF2()(f, m1, m2, chi1, chi2)
    assert hp_batch.shape == (B, N_f)
    assert hc_batch.shape == (B, N_f)

    for b in range(B):
        hp_single, hc_single = TaylorF2()(f, m1[b], m2[b], chi1[b], chi2[b])
        assert torch.allclose(hp_batch[b], hp_single, atol=1e-12)
        assert torch.allclose(hc_batch[b], hc_single, atol=1e-12)


def test_taylorf2_lal_parity():
    """Verify match > 0.995 against LALSuite TaylorF2 for spinning systems."""
    try:
        import lal
        import lalsimulation as lalsim
        from pycbc.filter.matchedfilter import match
        from pycbc.types import FrequencySeries
    except ImportError:
        pytest.skip("LALSuite or PyCBC not installed in current environment")

    test_cases = [
        # (m1, m2, chi1, chi2, f_min, f_max, min_match)
        (1.4, 1.4, 0.0, 0.0, 30.0, 1000.0, 0.999),
        (1.4, 1.4, 0.2, 0.1, 30.0, 1000.0, 0.998),
        (1.4, 1.4, 0.4, 0.3, 30.0, 1000.0, 0.995),
        (10.0, 10.0, 0.0, 0.0, 30.0, 200.0, 0.999),
        (20.0, 10.0, 0.3, -0.2, 30.0, 200.0, 0.994),
    ]

    df = 0.25
    for m1_val, m2_val, chi1_val, chi2_val, f_min, f_max, min_match in test_cases:
        extra = lal.CreateDict()
        hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
            m1_val * lal.MSUN_SI, m2_val * lal.MSUN_SI,
            0.0, 0.0, chi1_val, 0.0, 0.0, chi2_val,
            100.0 * 1e6 * lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0,
            df, f_min, f_max, f_min, extra, lalsim.TaylorF2
        )
        lal_series = FrequencySeries(hp_lal.data.data, delta_f=df)
        freqs = torch.arange(len(lal_series), dtype=torch.float64) * df

        hp_tw, _ = get_fd_waveform(
            "TaylorF2",
            mass1=m1_val,
            mass2=m2_val,
            chi1=chi1_val,
            chi2=chi2_val,
            sample_frequencies=freqs,
            distance=100.0,
            inclination=0.0,
            f_isco_cutoff=False,
        )

        tw_series = FrequencySeries(hp_tw.detach().cpu().numpy(), delta_f=df)
        overlap, _ = match(lal_series, tw_series, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
        assert overlap >= min_match, f"Match {overlap:.6f} below expected {min_match} for {m1_val},{m2_val} spins {chi1_val},{chi2_val}"


@pytest.mark.parametrize("model_name", ["TaylorF2", "IMRPhenomD", "TaylorF2_Eccentric"])
def test_spin_alias_routing(model_name):
    """Verify spin1z and chi1 are consistent aliases and reject conflicts."""
    f = torch.arange(20.0, 300.0, 0.5, dtype=torch.float64)
    cfg = dict(mass1=30.0, mass2=20.0, f_ref=30.0, sample_frequencies=f)

    # Base zero-spin waveform
    h_zero = get_fd_waveform(model_name, **cfg)[0]

    # Passing spin1z should produce spinning waveform
    h_s1z = get_fd_waveform(model_name, **cfg, spin1z=0.3)[0]
    assert not torch.equal(h_zero, h_s1z), f"{model_name}: spin1z=0.3 was ignored"

    # Passing chi1 should produce identical spinning waveform to exact equality
    h_chi1 = get_fd_waveform(model_name, **cfg, chi1=0.3)[0]
    assert torch.equal(h_s1z, h_chi1), f"{model_name}: spin1z and chi1 do not match"

    # Conflicting definitions must raise ValueError
    with pytest.raises(ValueError, match="Conflicting spin1"):
        get_fd_waveform(model_name, **cfg, chi1=0.3, spin1z=0.1)

    # Explicit zero on chi1 with non-zero spin1z must raise ValueError
    with pytest.raises(ValueError, match="Conflicting spin1"):
        get_fd_waveform(model_name, **cfg, chi1=0.0, spin1z=0.3)

    # Unknown keywords must be rejected
    with pytest.raises((ValueError, TypeError)):
        get_fd_waveform(model_name, **cfg, misspelled_spin=0.3)


@pytest.mark.parametrize("model_name", ["TaylorF2", "IMRPhenomD", "TaylorF2_Eccentric"])
def test_spin_alias_batch_routing(model_name):
    """Verify batched spin aliases broadcast and resolve without tensor boolean ambiguity."""
    f = torch.arange(20.0, 300.0, 0.5, dtype=torch.float64)
    cfg = dict(mass1=torch.tensor([30.0, 30.0]), mass2=torch.tensor([20.0, 20.0]), f_ref=30.0, sample_frequencies=f)
    spins = torch.tensor([0.1, 0.2])

    hp, hc = get_fd_waveform(model_name, **cfg, chi1=spins, spin1z=spins)
    assert hp.shape == (2, len(f))
    assert hc.shape == (2, len(f))


@pytest.mark.parametrize("model_name", ["TaylorF2", "IMRPhenomD", "TaylorF2_Eccentric"])
def test_aligned_in_plane_spin_rejection(model_name):
    """Verify aligned-spin models explicitly reject non-zero in-plane spins."""
    f = torch.arange(20.0, 200.0, 1.0, dtype=torch.float64)
    cfg = dict(mass1=30.0, mass2=20.0, sample_frequencies=f)

    with pytest.raises(ValueError, match="only supports aligned spins"):
        get_fd_waveform(model_name, **cfg, spin1x=0.1)

    with pytest.raises(ValueError, match="only supports aligned spins"):
        get_fd_waveform(model_name, **cfg, spin2y=0.2)
