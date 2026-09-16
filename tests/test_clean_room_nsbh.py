"""NSBH equation/invariant checks, without waveform-oracle imports.

Primary sources and unresolved reconstruction gaps are in
docs/reviews/2026-09-12-clean-room/nsbh.md.
"""

import pytest
import torch

from torchwave.approximants.imrphenom_nsbh import (
    IMRPhenomNSBH,
    _amplitude_window,
    ns_compactness_from_lambda,
)
from torchwave.constants import MSUN_SEC
from torchwave.approximants.nsbh_clean_phase import nrtidalv2_frequency_phase


def test_amplitude_window_width_is_in_total_mass_units():
    # At a separation of d/4 in M*f, Eq. (30) gives (1 +/- tanh(1))/2.
    mass_sec = torch.tensor([[6.4], [8.4]], dtype=torch.float64) * MSUN_SEC
    width_mf = torch.tensor([[0.015], [0.065]], dtype=torch.float64)
    center_mf = torch.tensor([[0.07], [0.09]], dtype=torch.float64)
    offsets = torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float64)
    frequencies = (center_mf + width_mf * offsets) / mass_sec
    rising = _amplitude_window(
        frequencies, center_mf / mass_sec, width_mf, mass_sec, rising=True
    )
    expected = torch.tensor(
        [0.11920292202211755, 0.5, 0.8807970779778824], dtype=torch.float64
    ).expand(2, -1)
    torch.testing.assert_close(rising, expected, rtol=2e-14, atol=2e-15)
    falling = _amplitude_window(
        frequencies, center_mf / mass_sec, width_mf, mass_sec, rising=False
    )
    torch.testing.assert_close(rising + falling, torch.ones_like(rising))


def test_compactness_black_hole_limit_and_c1_join():
    lam = torch.tensor([0.0, 1.0 - 1e-9, 1.0 + 1e-9], dtype=torch.float64, requires_grad=True)
    compactness = ns_compactness_from_lambda(lam)
    slopes = torch.autograd.grad(compactness.sum(), lam)[0]
    assert compactness[0].item() == 0.5
    torch.testing.assert_close(compactness[1:], torch.full((2,), 0.360, dtype=torch.float64), atol=1e-9, rtol=0)
    torch.testing.assert_close(slopes[1:], torch.full((2,), -0.0355, dtype=torch.float64), atol=3e-9, rtol=0)


def test_nsbh_amplitude_batch_matches_scalars_and_has_finite_gradients():
    model = IMRPhenomNSBH()
    f = torch.tensor([30.0, 200.0, 700.0, 1200.0, 2000.0], dtype=torch.float64)
    m1 = torch.tensor([5.0, 7.0], dtype=torch.float64)
    m2 = torch.tensor([1.4, 1.4], dtype=torch.float64)
    spin = torch.tensor([0.0, 0.4], dtype=torch.float64, requires_grad=True)
    lam = torch.tensor([800.0, 2600.0], dtype=torch.float64, requires_grad=True)
    batched = model(f, m1, m2, spin, lambda2=lam, inclination=0.9, f_ref=40.0)
    for polarization in batched:
        assert polarization.shape == (2, 5)
        assert torch.isfinite(polarization).all()
    for index in range(2):
        scalar = model(f, m1[index], m2[index], spin[index], lambda2=lam[index], inclination=0.9, f_ref=40.0)
        for full, single in zip(batched, scalar):
            torch.testing.assert_close(full[index], single, rtol=1e-9, atol=1e-35)
    gradients = torch.autograd.grad((batched[0].abs() * 1e22).sum(), (spin, lam))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_float32_scalar_and_batch_keep_one_precision_path():
    """The default NSBH gauge must not promote only batched parameters to float64."""
    f = torch.linspace(30.0, 500.0, 48, dtype=torch.float32)
    model = IMRPhenomNSBH()

    hp_scalar, hc_scalar = model(f, mass1=5.0, mass2=1.4, chi1=0.35, lambda2=450.0)
    hp_batch, hc_batch = model(
        f,
        mass1=torch.tensor([5.0], dtype=torch.float32),
        mass2=torch.tensor([1.4], dtype=torch.float32),
        chi1=torch.tensor([0.35], dtype=torch.float32),
        lambda2=torch.tensor([450.0], dtype=torch.float32),
    )

    assert hp_scalar.dtype == torch.complex64
    assert hc_scalar.dtype == torch.complex64
    assert hp_batch.dtype == torch.complex64
    assert hc_batch.dtype == torch.complex64
    for batched, scalar in ((hp_batch[0], hp_scalar), (hc_batch[0], hc_scalar)):
        torch.testing.assert_close(batched, scalar, rtol=2e-5, atol=0)
        # Accumulate in double precision: squaring physical strain amplitudes
        # (and especially their differences) can underflow in float32.
        delta = batched.to(torch.complex128) - scalar.to(torch.complex128)
        relative_norm = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(scalar.to(torch.complex128))
        assert relative_norm <= 2e-5


def test_nsbh_shared_frequencies_do_not_depend_on_requested_grid_extent():
    model = IMRPhenomNSBH()
    frequencies = torch.tensor([40.0, 80.0, 200.0, 800.0], dtype=torch.float64)
    extended = torch.cat((frequencies, frequencies.new_tensor([1600.0, 2400.0])))
    parameters = dict(mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0, f_ref=40.0, phasing_convention="reference_frequency")
    original = model(frequencies, **parameters)
    larger = model(extended, **parameters)
    different_bound = model(frequencies, f_final=3000.0, **parameters)
    for before, after, bounded in zip(original, larger, different_bound):
        torch.testing.assert_close(before, after[:frequencies.numel()], rtol=2e-13, atol=1e-35)
        torch.testing.assert_close(before, bounded, rtol=2e-13, atol=1e-35)


@pytest.mark.parametrize("reference_mf", [0.008, 0.025, 0.08])
def test_nsbh_zero_tidal_reference_phase_in_each_phenomd_region(reference_mf):
    # With Lambda=0 the intrinsic phase at f_ref cancels in every IMR region.
    # Retain the NSBH carrier's overall minus sign and the orbital m=2 phase.
    reference_hz = reference_mf / (8.4 * MSUN_SEC)
    frequencies = torch.tensor([reference_hz], dtype=torch.float64)
    phase = 0.71
    hp, _ = IMRPhenomNSBH()(
        frequencies, 7.0, 1.4, chi1=0.4, lambda2=0.0,
        phic=phase, f_ref=reference_hz, phasing_convention="reference_frequency",
    )
    expected = -torch.exp(torch.tensor(2j * phase, dtype=torch.complex128))
    torch.testing.assert_close(hp / hp.abs(), expected.expand_as(hp), rtol=2e-12, atol=2e-12)


def test_nsbh_time_and_orbital_phase_translation():
    frequencies = torch.tensor([40.0, 80.0, 200.0, 800.0], dtype=torch.float64)
    parameters = dict(mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0, f_ref=40.0, phasing_convention="reference_frequency")
    model = IMRPhenomNSBH()
    baseline = model(frequencies, **parameters)
    shifted = model(frequencies, tc=0.013, phic=0.37, **parameters)
    expected = torch.exp(1j * (2.0 * 0.37 - 2.0 * torch.pi * frequencies * 0.013))
    for original, translated in zip(baseline, shifted):
        torch.testing.assert_close(translated / original, expected, rtol=2e-11, atol=2e-11)


def test_nsbh_coalescence_convention_and_alias_ignore_reference_frequency():
    frequencies = torch.tensor([40.0, 200.0, 800.0], dtype=torch.float64)
    parameters = dict(mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0)
    model = IMRPhenomNSBH()
    first = model(frequencies, f_ref=40.0, phasing_convention="coalescence", **parameters)
    second = model(frequencies, f_ref=800.0, reference_frame="coalescence", **parameters)
    for low_reference, high_reference in zip(first, second):
        torch.testing.assert_close(low_reference, high_reference, rtol=2e-13, atol=1e-35)


def test_nsbh_tidal_phase_is_additive_and_linear_in_deformability():
    # Dietrich Eq. (18) is the additive tidal phase callable itself.  Test
    # that public equation directly; the complete NSBH waveform also updates
    # its remnant spin with Lambda and therefore is not linear in Lambda.
    frequencies = torch.tensor([50.0, 300.0, 800.0], dtype=torch.float64)
    total_mass = torch.tensor(8.4 * MSUN_SEC, dtype=torch.float64)
    eta = torch.tensor(7.0 * 1.4 / 8.4**2, dtype=torch.float64)
    phases = []
    for deformability in (0.0, 1200.0, 2400.0):
        phases.append(nrtidalv2_frequency_phase(frequencies, total_mass, eta, deformability))
    torch.testing.assert_close(2 * phases[1], phases[0] + phases[2], rtol=2e-12, atol=2e-12)


def test_nsbh_default_retains_legacy_gauge_and_reference_centered_time_shift():
    frequencies = torch.tensor([40.0, 80.0, 200.0, 800.0], dtype=torch.float64)
    parameters = dict(mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0, f_ref=40.0)
    model = IMRPhenomNSBH()
    default = model(frequencies, **parameters)
    named = model(frequencies, phasing_convention="legacy", **parameters)
    shifted = model(frequencies, tc=0.013, phic=0.37, **parameters)
    expected = torch.exp(1j * (2.0 * 0.37 + 2.0 * torch.pi * (frequencies - 40.0) * 0.013))
    for initial, legacy, translated in zip(default, named, shifted):
        torch.testing.assert_close(initial, legacy, rtol=0, atol=0)
        torch.testing.assert_close(translated / initial, expected, rtol=2e-11, atol=2e-11)


def test_nsbh_zero_reference_uses_lowest_positive_frequency():
    frequencies = torch.tensor([0.0, 20.0, 40.0, 200.0], dtype=torch.float64)
    parameters = dict(mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0)
    model = IMRPhenomNSBH()

    zero = model(frequencies, f_ref=0.0, **parameters)
    explicit = model(frequencies, f_ref=20.0, **parameters)
    default = model(frequencies, **parameters)
    default_explicit = model(frequencies, f_ref=30.0, **parameters)
    for resolved, expected in zip(zero, explicit):
        assert torch.isfinite(resolved).all()
        torch.testing.assert_close(resolved, expected, rtol=2e-13, atol=1e-35)
    for omitted, explicit_default in zip(default, default_explicit):
        torch.testing.assert_close(omitted, explicit_default, rtol=0, atol=0)


@pytest.mark.parametrize("bad_reference", [-1.0, float("nan"), float("inf")])
def test_nsbh_negative_or_nonfinite_reference_is_rejected(bad_reference):
    frequencies = torch.tensor([0.0, 20.0, 40.0], dtype=torch.float64)
    parameters = dict(mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0)
    with pytest.raises(ValueError, match="f_ref"):
        IMRPhenomNSBH()(frequencies, f_ref=bad_reference, **parameters)


def test_nsbh_zero_reference_on_an_all_nonpositive_grid_stays_finite():
    frequencies = torch.tensor([-20.0, 0.0, -1.0], dtype=torch.float64)
    parameters = dict(mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0)
    hp, hc = IMRPhenomNSBH()(frequencies, f_ref=0.0, **parameters)
    assert torch.equal(hp, torch.zeros_like(hp))
    assert torch.equal(hc, torch.zeros_like(hc))


def test_nsbh_zero_reference_batched_parameters_use_shared_grid_sentinel():
    frequencies = torch.tensor([0.0, -5.0, 20.0, 80.0], dtype=torch.float64)
    parameters = dict(
        mass1=torch.tensor([7.0, 8.0], dtype=torch.float64),
        mass2=torch.tensor([1.4, 1.3], dtype=torch.float64),
        chi1=torch.tensor([0.4, 0.1], dtype=torch.float64),
        lambda2=torch.tensor([2600.0, 400.0], dtype=torch.float64),
    )
    model = IMRPhenomNSBH()
    batched = model(frequencies, f_ref=0.0, **parameters)
    for index in range(2):
        scalar = model(
            frequencies,
            f_ref=20.0,
            mass1=parameters["mass1"][index],
            mass2=parameters["mass2"][index],
            chi1=parameters["chi1"][index],
            lambda2=parameters["lambda2"][index],
        )
        for actual, expected in zip(batched, scalar):
            torch.testing.assert_close(actual[index], expected, rtol=2e-13, atol=1e-35)


def test_nsbh_reference_sentinel_preserves_finite_frequency_gradients():
    frequencies = torch.tensor([0.0, 20.0, 40.0], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    hp, hc = IMRPhenomNSBH()(
        frequencies,
        mass1=7.0,
        mass2=1.4,
        chi1=0.4,
        lambda2=2600.0,
        f_ref=reference,
        phasing_convention="reference_frequency",
    )
    gradients = torch.autograd.grad((hp.abs() + hc.abs()).sum(), (frequencies, reference))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_nsbh_gauges_change_only_affine_phase_and_fixed_anchor_is_grid_independent():
    frequencies = torch.linspace(50.0, 2400.0, 41, dtype=torch.float64)
    extended = torch.cat((frequencies, frequencies.new_tensor([2800.0])))
    parameters = dict(mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0, f_ref=40.0)
    model = IMRPhenomNSBH()
    legacy = model(frequencies, f_final=3000.0, **parameters)
    larger = model(extended, f_final=3000.0, **parameters)
    explicit = model(frequencies, phasing_convention="reference_frequency", **parameters)
    for old, new, consistent in zip(legacy, explicit, larger):
        torch.testing.assert_close(old, consistent[:frequencies.numel()], rtol=2e-13, atol=1e-35)
        ratio = old / new
        torch.testing.assert_close(ratio.abs(), torch.ones_like(ratio.abs()), rtol=2e-14, atol=2e-14)
        # The second difference of an affine phase is zero, including across
        # the inspiral/intermediate/merger joins in the underlying phase.
        curvature = ratio[2:] * ratio[:-2] / ratio[1:-1].square()
        torch.testing.assert_close(curvature, torch.ones_like(curvature), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("convention", ["legacy", "reference_frequency"])
def test_nsbh_reference_frequency_changes_only_phase_gauge(convention):
    # The published PhenomD construction permits a linear time/phase shift;
    # changing f_ref must therefore leave intrinsic curvature and amplitude
    # unchanged.  NRTidal is added as an intrinsic phase correction and does
    # not alter this affine identity.
    frequencies = torch.linspace(40.0, 900.0, 19, dtype=torch.float64)
    model = IMRPhenomNSBH()
    common = dict(
        mass1=7.0, mass2=1.4, chi1=0.4, lambda2=2600.0,
        phasing_convention=convention,
    )
    low = model(frequencies, f_ref=30.0, **common)
    high = model(frequencies, f_ref=80.0, **common)
    for first, second in zip(low, high):
        ratio = second / first
        torch.testing.assert_close(ratio.abs(), torch.ones_like(ratio.abs()), rtol=2e-14, atol=2e-14)
        curvature = ratio[2:] * ratio[:-2] / ratio[1:-1].square()
        torch.testing.assert_close(curvature, torch.ones_like(curvature), rtol=3e-12, atol=3e-12)
