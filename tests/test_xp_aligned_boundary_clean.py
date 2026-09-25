"""Clean-room regressions for the XP v223 aligned chart endpoint."""

import math

import lal
import lalsimulation as ls
import pytest
import torch

from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.precession_msa_waveform_clean import waveform_v223
from torchwave.approximants.precession_source_frame_clean import source_frame_geometry


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


@pytest.mark.parametrize("phic", (0.0, 0.3, 0.7))
@pytest.mark.parametrize("spin1x", (0.0, 1.0e-9))
def test_endpoint_combined_angle_is_chart_independent(phic, spin1x):
    # Appendix C Eqs. (67), (75), and (79) fix the beta->0 quadrupole
    # projection through exp[2i(alpha0+gamma0)], even when phi_J is undefined
    # at exact alignment.  The raw exact chart can differ by pi in alpha0.
    alpha0, _theta, _zeta, phi_j = source_frame_geometry(
        35.0, 25.0, spin1x, 0.0, 0.4, 0.0, 0.0, -0.2,
        0.7, phic, 20.0, compatibility="v223",
    )
    combined = torch.exp(2j * (alpha0 + math.pi - phi_j))
    expected = torch.exp(torch.tensor(2j * (1.5 * math.pi + phic), dtype=torch.complex128))
    torch.testing.assert_close(combined, expected, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("convention", (0, 1))
def test_audited_exact_aligned_matches_public_boundary(convention):
    actual = _clean_waveform(convention)
    target = _public_target(convention)
    assert _max_relative_l2(actual, target) < STRICT_GATE


@pytest.mark.parametrize("convention", (0, 1))
def test_generic_table_iv_combined_rotation_matches_public_oracle(convention):
    # Table IV's carrier phase is only one part of the final strain; the
    # observable includes alpha, gamma, and the Appendix C output basis.
    # Compare that combined result with the public binary well away from
    # the singular J_perp=0 chart.
    source = dict(spin1=(0.3, 0.2, 0.4), spin2=(-0.2, 0.1, -0.2))
    actual = _clean_waveform(convention, **source)
    target = _public_target(convention, **source)
    assert _max_relative_l2(actual, target) < STRICT_GATE


def test_v223_reference_closure_matches_public_lal_residual():
    # Appendix C Eq. (79b) identifies beta_0 with the geometric J-L angle.
    # The public v223 implementation uses distinct orbital-L coefficients for
    # source-frame geometry and MSA opening beta. Preserve that binary
    # comparison in the explicit v223 helper; the public one-spin path uses
    # the paper-derived physical opening.
    f_ref = 40.0
    frequency = torch.tensor([f_ref], dtype=DTYPE)
    xp = IMRPhenomXP()
    native_xp = xp(
        frequency, mass1=40.0, mass2=20.0,
        spin1x=0.3, spin1y=0.2, spin1z=0.4, spin2z=-0.2,
        distance=100.0, inclination=0.7, phic=0.3, f_ref=f_ref,
        prec_version=223, final_spin_mod=3, convention=1,
    )
    remnant_overrides = []

    def recording_carrier(**kwargs):
        remnant_overrides.append(kwargs["_final_spin_override"])
        return xp.xas_model(**kwargs)

    clean_xp = waveform_v223(
        frequency, 40.0, 20.0,
        spin1x=0.3, spin1y=0.2, spin1z=0.4, spin2z=-0.2,
        distance=100.0, inclination=0.7, phic=0.3, f_ref=f_ref,
        convention=1, final_spin_mod=3, xas_model=recording_carrier,
    )
    physical_xp = waveform_v223(
        frequency, 40.0, 20.0,
        spin1x=0.3, spin1y=0.2, spin1z=0.4, spin2z=-0.2,
        distance=100.0, inclination=0.7, phic=0.3, f_ref=f_ref,
        convention=1, final_spin_mod=3, boundary="physical",
    )
    for actual, physical in zip(native_xp, physical_xp):
        torch.testing.assert_close(actual, physical, rtol=2e-13, atol=0.0)
    assert len(remnant_overrides) == 1
    carrier_kwargs = dict(
        f=frequency, mass1=40.0, mass2=20.0, chi1=0.4, chi2=-0.2,
        distance=100.0, inclination=0.7, phi_c=0.3, f_ref=f_ref,
    )
    native_carrier = xp.xas_model(**carrier_kwargs)
    remnant_carrier = xp.xas_model(
        **carrier_kwargs, _final_spin_override=remnant_overrides[0]
    )
    for ordinary, matched in zip(native_carrier, remnant_carrier):
        # At the reference frequency, the XAS carrier is insensitive to the
        # remnant override. The remaining XP/XAS ratio isolates frame closure.
        torch.testing.assert_close(ordinary, matched, rtol=2e-13, atol=0.0)

    def public_waveform(approximant, spin1, spin2, options):
        hp, hc = ls.SimInspiralChooseFDWaveform(
            40.0 * lal.MSUN_SI, 20.0 * lal.MSUN_SI,
            *spin1, *spin2, 100.0e6 * lal.PC_SI, 0.7, 0.3,
            0.0, 0.0, 0.0, 1.0, 20.0, 64.0, f_ref,
            options, approximant,
        )
        return hp.data.data[40], hc.data.data[40]

    public_xp = public_waveform(
        ls.IMRPhenomXP, (0.3, 0.2, 0.4), (0.0, 0.0, -0.2), _options(1)
    )
    public_carrier = public_waveform(
        ls.IMRPhenomXAS, (0.0, 0.0, 0.4), (0.0, 0.0, -0.2),
        lal.CreateDict(),
    )
    for wave, carrier, oracle_wave, oracle_carrier in zip(
        clean_xp, remnant_carrier, public_xp, public_carrier
    ):
        torch.testing.assert_close(
            carrier[0], torch.as_tensor(oracle_carrier), rtol=2e-12, atol=0.0
        )
        native_ratio = wave[0] / carrier[0]
        oracle_ratio = oracle_wave / oracle_carrier
        torch.testing.assert_close(
            native_ratio, torch.as_tensor(oracle_ratio), rtol=0.0, atol=2e-12
        )


@pytest.mark.parametrize(
    "spin1,spin2,phic",
    (
        ((0.3, 0.2, 0.4), (-0.2, 0.1, -0.2), -0.7),
        ((0.0, 0.0, 0.4), (0.0, 0.0, -0.2), 0.3),
    ),
)
def test_public_convention6_uses_clean_quadrupole_phase(spin1, spin2, phic):
    # Pratten et al. Appendix F Table IV changes the orbital-phase carrier
    # between conventions 1 and 6. The full output includes the Appendix C
    # rotation and polarization basis, so compare both polarizations with
    # the public v223 binary in generic and aligned source-frame charts.
    p = dict(
        mass1=35.0, mass2=25.0,
        spin1x=spin1[0], spin1y=spin1[1], spin1z=spin1[2],
        spin2x=spin2[0], spin2y=spin2[1], spin2z=spin2[2],
        distance=500.0, inclination=0.7, phic=phic, f_ref=20.0,
    )
    model = IMRPhenomXP()
    convention1 = model(FREQUENCIES, **p, convention=1)
    convention6 = model(FREQUENCIES, **p, convention=6)
    phase = torch.exp(torch.tensor(2j * phic, dtype=torch.complex128))
    for actual, base in zip(convention6, convention1):
        torch.testing.assert_close(actual, phase * base, rtol=1e-14, atol=1e-35)
    target = _public_target(
        6, mass1=35.0, mass2=25.0, spin1=spin1, spin2=spin2, phic=phic,
    )
    if spin1[:2] == spin2[:2] == (0.0, 0.0):
        binary = _clean_waveform(1, mass1=35.0, mass2=25.0,
                                 spin1=spin1, spin2=spin2, phic=phic)
        binary6 = tuple(phase * h for h in binary)
        assert _max_relative_l2(binary6, target) < STRICT_GATE
        # The public physical endpoint also enforces the paper's beta=0
        # minimal-rotation law. The v223 compatibility chart retains an
        # aligned alpha+gamma drift, so its LAL waveform remains distinct.
        assert _max_relative_l2(convention6, target) > 0.05
    else:
        assert _max_relative_l2(convention6, target) < STRICT_GATE


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
            batched[0][row], expected[0], rtol=1e-15, atol=1e-35
        )
        torch.testing.assert_close(
            batched[1][row], expected[1], rtol=1e-15, atol=1e-35
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
