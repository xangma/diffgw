"""Primary-equation checks for the NNLO precessing remnant plumbing."""

import pytest
import torch

from torchwave.approximants.imrphenom_xas import XASParameterFits
from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.approximants.precession_msa import compute_effective_chip


DTYPE = torch.float64


def t(value, *, requires_grad=False):
    return torch.tensor(value, dtype=DTYPE, requires_grad=requires_grad)


@pytest.mark.parametrize("symmetric_mass_ratio", [0.24, 0.25])
def test_nnlo_final_spin_matches_equations_and_aligned_limit(symmetric_mass_ratio):
    eta = t([symmetric_mass_ratio])
    chi1, chi2, chip = t([0.4]), t([-0.2]), t([0.3])
    aligned = XASParameterFits.fit_final_spin(eta, chi1, chi2)
    m_big_over_m = 0.5 * (1.0 + torch.sqrt(1.0 - 4.0 * eta))
    expected = torch.sign(aligned) * torch.sqrt(
        aligned.square() + (chip * m_big_over_m.square()).square()
    )
    actual = XASParameterFits.fit_final_spin_nnlo(eta, chi1, chi2, chip)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-14)
    torch.testing.assert_close(
        XASParameterFits.fit_final_spin_nnlo(eta, chi1, chi2, torch.zeros_like(chip)),
        aligned,
        atol=0.0, rtol=0.0,
    )


def test_nnlo_final_spin_rejects_undefined_zero_parallel_sign(monkeypatch):
    def zero_aligned_fit(eta, chi1, chi2):
        return torch.zeros_like(eta)

    monkeypatch.setattr(XASParameterFits, "fit_final_spin", staticmethod(zero_aligned_fit))
    with pytest.raises(ValueError, match="sign is undefined"):
        XASParameterFits.fit_final_spin_nnlo(t([0.24]), t([0.4]), t([-0.2]), t([0.3]))


def test_nnlo_signed_negative_and_tiny_magnitude(monkeypatch):
    def tiny_negative_fit(eta, chi1, chi2):
        return torch.full_like(eta, -1.0e-8)

    monkeypatch.setattr(XASParameterFits, "fit_final_spin", staticmethod(tiny_negative_fit))
    result = XASParameterFits.fit_final_spin_nnlo(
        t([0.24]), t([0.4]), t([-0.2]), t([1.0e-8])
    )
    expected = -(1.0 + (0.6**2) ** 2) ** 0.5 * 1.0e-8
    torch.testing.assert_close(result, t([expected]), atol=1e-16, rtol=1e-12)


def test_nnlo_chip_mass_body_and_in_plane_rotation_invariants():
    m1, m2 = t(30.0), t(20.0)
    spins = (t(0.3), t(0.2), t(-0.2), t(0.1))
    direct = compute_effective_chip(m1, m2, *spins)
    scaled = compute_effective_chip(2.0 * m1, 2.0 * m2, *spins)
    angle = t(0.73)
    c, s = torch.cos(angle), torch.sin(angle)
    rotated = (c * spins[0] - s * spins[1], s * spins[0] + c * spins[1],
               c * spins[2] - s * spins[3], s * spins[2] + c * spins[3])
    torch.testing.assert_close(scaled, direct)
    torch.testing.assert_close(compute_effective_chip(m1, m2, *rotated), direct)

    equal_mass = compute_effective_chip(
        t(30.0), t(30.0), spins[0], spins[1], spins[2], spins[3]
    )
    torch.testing.assert_close(equal_mass, torch.maximum(torch.hypot(spins[0], spins[1]), torch.hypot(spins[2], spins[3])))


@pytest.mark.parametrize("model", [IMRPhenomXP, IMRPhenomXPHM])
def test_nnlo_waveform_is_invariant_under_body_relabeling(model):
    frequency = t([25.0, 80.0, 180.0])
    bodies = dict(mass1=40.0, mass2=20.0, spin1x=0.3, spin1y=0.2,
                  spin1z=0.1, spin2x=-0.2, spin2y=0.3, spin2z=-0.1)
    swapped = dict(mass1=bodies["mass2"], mass2=bodies["mass1"])
    for axis in "xyz":
        swapped[f"spin1{axis}"] = bodies[f"spin2{axis}"]
        swapped[f"spin2{axis}"] = bodies[f"spin1{axis}"]
    instance = model()
    for version in (101, 102, 103, 104):
        options = dict(inclination=0.7, phic=0.3, f_ref=40.0, prec_version=version)
        original = instance(frequency, **bodies, **options)
        relabeled = instance(frequency, **swapped, **options)
        for actual, expected in zip(relabeled, original):
            torch.testing.assert_close(actual, expected, atol=0.0, rtol=1e-13)


@pytest.mark.parametrize("model", [IMRPhenomXP, IMRPhenomXPHM])
def test_nnlo_mixed_batch_zero_in_plane_row_has_finite_gradients_and_aligned_limit(model):
    frequency = t([20.0, 40.0])
    spin1x = t([0.0, 0.20], requires_grad=True)
    spin1y = t([0.0, 0.10], requires_grad=True)
    spin2x = t([0.0, 0.05])
    spin2y = t([0.0, 0.02])
    kwargs = dict(
        mass1=t([30.0, 30.0]),
        mass2=t([20.0, 20.0]),
        spin1x=spin1x,
        spin1y=spin1y,
        spin1z=t([0.4, 0.4]),
        spin2x=spin2x,
        spin2y=spin2y,
        spin2z=t([-0.2, -0.2]),
    )
    hp, hc = model()(f=frequency, **kwargs)
    loss = hp.real.sum() + hc.real.sum()
    grad_x, grad_y = torch.autograd.grad(loss, (spin1x, spin1y))
    assert torch.isfinite(grad_x).all()
    assert torch.isfinite(grad_y).all()

    aligned_kwargs = dict(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.4,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.2,
    )
    aligned = model()(f=frequency, **aligned_kwargs)
    torch.testing.assert_close(hp[0], aligned[0], atol=0.0, rtol=2e-6)
    torch.testing.assert_close(hc[0], aligned[1], atol=0.0, rtol=2e-6)


def test_nnlo_chip_is_zero_and_differentiable_at_aligned_spin():
    x = t([0.0], requires_grad=True)
    y = t([0.0], requires_grad=True)
    u = t([0.0], requires_grad=True)
    v = t([0.0], requires_grad=True)
    chip = compute_effective_chip(t([30.0]), t([20.0]), x, y, u, v)
    gradients = torch.autograd.grad(chip.sum(), (x, y, u, v))
    assert torch.equal(chip, torch.zeros_like(chip))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("model", [IMRPhenomXP, IMRPhenomXPHM])
def test_nnlo_version_dispatch_keeps_msa_explicit_versions_distinct(model):
    frequency = t([25.0, 40.0, 80.0])
    kwargs = dict(
        mass1=40.0, mass2=20.0, spin1x=0.3, spin1y=0.2, spin1z=0.1,
        spin2x=-0.2, spin2y=0.3, spin2z=-0.1, inclination=0.7, phic=0.3,
        f_ref=40.0,
    )
    instance = model() if model is IMRPhenomXP else model(modes=[(2, 2)])
    default = instance(frequency, **kwargs)
    explicit_nnlo = instance(frequency, **kwargs, prec_version=102)
    method_nnlo = instance(frequency, **kwargs, precession_method="nnlo", prec_version=102)
    method_msa_nnlo = instance(frequency, **kwargs, precession_method="msa", prec_version=102)
    method_msa = instance(frequency, **kwargs, precession_method="msa")
    explicit_msa = instance(frequency, **kwargs, prec_version=223)
    for expected in (explicit_nnlo, method_nnlo, method_msa_nnlo):
        for actual, reference in zip(expected, default):
            torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
    for actual, reference in zip(method_msa, explicit_msa):
        torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
