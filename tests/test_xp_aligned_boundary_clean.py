"""Clean-room regressions for the XP v223 aligned chart endpoint."""

import math

import lal
import lalsimulation as ls
import pytest
import torch

from torchwave.approximants.precession_msa_waveform_clean import waveform_v223


DTYPE = torch.float64
FREQUENCIES = torch.arange(20.0, 512.0, 0.25, dtype=DTYPE)
STRICT_GATE = 1.0e-6


def _options(convention):
    params = lal.CreateDict()
    ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(params, 223)
    ls.SimInspiralWaveformParamsInsertPhenomXPExpansionOrder(params, 5)
    ls.SimInspiralWaveformParamsInsertPhenomXPConvention(params, convention)
    ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(params, 3)
    return params


def _public_target(
    convention,
    *,
    mass1=35.0,
    mass2=25.0,
    spin1=(0.0, 0.0, 0.2),
    spin2=(0.0, 0.0, -0.1),
    phic=0.3,
):
    """Evaluate the public XP binary on the clean quarter-Hz grid."""
    hp, hc = ls.SimInspiralChooseFDWaveform(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        *spin1,
        *spin2,
        500.0e6 * lal.PC_SI,
        0.7,
        phic,
        0.0,
        0.0,
        0.0,
        0.25,
        20.0,
        512.0,
        20.0,
        _options(convention),
        ls.IMRPhenomXP,
    )
    # The public uniform series starts at 0 Hz; the clean grid is 20 <= f <
    # 512 Hz.
    return (
        torch.as_tensor(hp.data.data[80:2048].copy(), dtype=torch.complex128),
        torch.as_tensor(hc.data.data[80:2048].copy(), dtype=torch.complex128),
    )


def _clean_waveform(
    convention,
    *,
    mass1=35.0,
    mass2=25.0,
    spin1=(0.0, 0.0, 0.2),
    spin2=(0.0, 0.0, -0.1),
    phic=0.3,
):
    return waveform_v223(
        FREQUENCIES,
        mass1,
        mass2,
        *spin1,
        *spin2,
        distance=500.0,
        inclination=0.7,
        phic=phic,
        tc=0.0,
        f_ref=20.0,
        convention=convention,
        final_spin_mod=3,
    )


def _max_relative_l2(actual, target):
    return max(
        float(torch.linalg.vector_norm(left - right)
               / torch.linalg.vector_norm(right))
        for left, right in zip(actual, target)
    )


@pytest.mark.parametrize("convention", (0, 1))
def test_audited_exact_aligned_matches_public_boundary(convention):
    actual = _clean_waveform(convention)
    target = _public_target(convention)
    assert _max_relative_l2(actual, target) < STRICT_GATE


@pytest.mark.parametrize("convention", (0, 1))
def test_zero_spin_and_fresh_aligned_fixture_match_public_boundary(convention):
    zero = dict(spin1=(0.0, 0.0, 0.0), spin2=(0.0, 0.0, 0.0))
    fresh = dict(
        mass1=45.0,
        mass2=15.0,
        spin1=(0.0, 0.0, 0.35),
        spin2=(0.0, 0.0, -0.2),
    )
    for case in (zero, fresh):
        actual = _clean_waveform(convention, **case)
        target = _public_target(convention, **case)
        assert _max_relative_l2(actual, target) < STRICT_GATE


@pytest.mark.parametrize(
    "phic", (0.0, -0.7, 2.0 * math.pi + 0.3, -2.0 * math.pi - 0.4)
)
def test_convention1_endpoint_chart_handles_zero_negative_and_wrapped_phic(phic):
    actual = _clean_waveform(1, phic=phic)
    target = _public_target(1, phic=phic)
    assert _max_relative_l2(actual, target) < STRICT_GATE


def test_exact_aligned_batch_matches_independent_scalar_rows():
    batch = dict(
        mass1=torch.tensor([35.0, 45.0], dtype=DTYPE),
        mass2=torch.tensor([25.0, 15.0], dtype=DTYPE),
        spin1x=torch.zeros(2, dtype=DTYPE),
        spin1y=torch.zeros(2, dtype=DTYPE),
        spin1z=torch.tensor([0.2, 0.35], dtype=DTYPE),
        spin2x=torch.zeros(2, dtype=DTYPE),
        spin2y=torch.zeros(2, dtype=DTYPE),
        spin2z=torch.tensor([-0.1, -0.2], dtype=DTYPE),
        distance=torch.tensor([500.0, 350.0], dtype=DTYPE),
        inclination=torch.tensor([0.7, 0.4], dtype=DTYPE),
        phic=torch.tensor([0.3, -0.7], dtype=DTYPE),
        tc=torch.tensor([0.0, 0.002], dtype=DTYPE),
        f_ref=torch.tensor([20.0, 30.0], dtype=DTYPE),
    )
    batched = waveform_v223(
        FREQUENCIES,
        **batch,
        convention=1,
        final_spin_mod=3,
    )
    assert batched[0].shape == (2, FREQUENCIES.numel())
    assert batched[1].shape == (2, FREQUENCIES.numel())
    for row in range(2):
        scalar = {name: value[row] for name, value in batch.items()}
        expected = waveform_v223(
            FREQUENCIES,
            **scalar,
            convention=1,
            final_spin_mod=3,
        )
        torch.testing.assert_close(
            batched[0][row], expected[0], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            batched[1][row], expected[1], rtol=0.0, atol=0.0
        )


def test_exact_aligned_endpoint_gradients_are_finite():
    mass1 = torch.tensor(35.0, dtype=DTYPE, requires_grad=True)
    phic = torch.tensor(-0.7, dtype=DTYPE, requires_grad=True)
    spin1x = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    spin1y = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    hp, hc = waveform_v223(
        torch.tensor([30.0, 80.0], dtype=DTYPE),
        mass1,
        25.0,
        spin1x,
        spin1y,
        0.2,
        0.0,
        0.0,
        -0.1,
        distance=500.0,
        inclination=0.7,
        phic=phic,
        tc=0.0,
        f_ref=20.0,
        convention=1,
        final_spin_mod=3,
    )
    loss = (hp.real.square() + hc.imag.square()).sum()
    gradients = torch.autograd.grad(loss, (mass1, phic, spin1x, spin1y))
    assert all(torch.isfinite(value) for value in gradients)


def test_tiny_nonzero_transverse_spin_is_not_endpoint_corrected():
    # This is intentionally below the legacy <=1e-28 squared chart selector.
    # The new mapping uses exact component equality, so this row remains on
    # the older near-aligned behavior rather than being treated as exact zero.
    tiny = _clean_waveform(
        1,
        spin1=(1.0e-15, 0.0, 0.2),
        spin2=(0.0, 0.0, -0.1),
    )
    exact = _clean_waveform(1)
    assert all(torch.isfinite(value).all() for value in tiny)
    assert any(
        torch.max(torch.abs(left - right)).item() > 0.0
        for left, right in zip(tiny, exact)
    )

