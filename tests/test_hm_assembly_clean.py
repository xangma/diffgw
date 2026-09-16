"""Meaningful equation-level tests for the clean HM assembly."""
import torch
import math
import pytest

from torchwave.approximants.hm_assembly_clean import (
    COMPATIBILITY_AMPLITUDE_JOIN,
    COMPATIBILITY_PHASE_JOIN,
    PRINTED_AMPLITUDE_JOIN,
    PRINTED_PHASE_JOIN,
    assemble_mode,
    project_positive_modes,
)
from torchwave.approximants.imrphenom_hm_phase_clean import (
    phenomd_eq14_phase_delta,
    phenomd_eq14_phase_delta_c1,
    phenomd_eq14_phase_delta_derivative,
)
from torchwave.approximants.precession_twisting_clean import wigner_d


def _kwargs():
    return dict(
        eta=0.21, delta=0.4, chi_s=0.05, chi_a=0.2,
        carrier_positive=lambda x: 0.7 + 3.0 * x + 0.2 * x**2,
        amplitude_22=lambda x: torch.ones_like(x),
        amp_qnm_22=0.086, amp_qnm_lm=0.14,
        phase_qnm_22=0.086, phase_qnm_lm=0.14,
        phase_damp_22=0.014, phase_damp_lm=0.02,
        alpha4=1.1, alpha5=0.7, chi_pn=0.03,
        reference_gauge=0.0, phic=0.0,
        # Preserve the former clean diagnostic recipe explicitly.
        amplitude_join=COMPATIBILITY_AMPLITUDE_JOIN,
        phase_join=COMPATIBILITY_PHASE_JOIN,
    )


def test_22_identity_preserves_public_carrier_phase():
    x = torch.tensor([0.01, 0.04, 0.09], dtype=torch.float64)
    kw = _kwargs()
    kw["phase_qnm_lm"] = kw["phase_qnm_22"]
    kw["phase_damp_lm"] = kw["phase_damp_22"]
    got = assemble_mode(x, (2, 2), **kw, use_eq14=True)
    expected = torch.exp(-1j * kw["carrier_positive"](x))
    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_eq14_distinct_hm_and_carrier_qnms_use_hm_ratios_only():
    x = torch.tensor([0.018, 0.043, 0.091], dtype=torch.float64, requires_grad=True)
    eta = 0.22
    chi_pn = 0.08
    hm_fr22, hm_flm = 0.083, 0.119
    hm_fd22, hm_fdlm = 0.014, 0.023
    carrier_fr22, carrier_fd22 = 0.061, 0.009
    alpha4, alpha5 = 1.13, 0.74
    ratio = hm_fr22 / hm_flm
    damping_ratio = hm_fdlm / hm_fd22
    expected = alpha4 / eta * (
        ratio * torch.atan((x - alpha5 * carrier_fr22) /
                           (carrier_fd22 * ratio * damping_ratio))
        - torch.atan((x - alpha5 * carrier_fr22) / carrier_fd22)
    )
    got = phenomd_eq14_phase_delta(
        x, eta, chi_pn, hm_fr22, hm_flm, hm_fd22, hm_fdlm,
        alpha4, alpha5, base_fr22=carrier_fr22, base_fd22=carrier_fd22,
    )
    assert torch.allclose(got, expected, atol=1e-14, rtol=1e-14)

    # Omitting the explicit carrier pair retains the helper's previous HM-22
    # algebra, including the original HM damping width.
    default = phenomd_eq14_phase_delta(
        x, eta, chi_pn, hm_fr22, hm_flm, hm_fd22, hm_fdlm, alpha4, alpha5,
    )
    legacy = alpha4 / eta * (
        ratio * torch.atan((x - alpha5 * hm_fr22) / (hm_fdlm * ratio))
        - torch.atan((x - alpha5 * hm_fr22) / hm_fd22)
    )
    assert torch.allclose(default, legacy, atol=1e-14, rtol=1e-14)

    analytic = phenomd_eq14_phase_delta_derivative(
        x, eta, chi_pn, hm_fr22, hm_flm, hm_fd22, hm_fdlm,
        alpha4, alpha5, base_fr22=carrier_fr22, base_fd22=carrier_fd22,
    )
    autodiff = torch.autograd.grad(got.sum(), x, create_graph=True)[0]
    assert torch.allclose(analytic, autodiff, atol=1e-13, rtol=1e-13)


def test_eq14_c1_uses_carrier_join_and_is_continuous_with_zero_slope():
    eta = 0.22
    chi_pn = 0.08
    hm_fr22, hm_flm = 0.083, 0.119
    hm_fd22, hm_fdlm = 0.014, 0.023
    carrier_fr22, carrier_fd22 = 0.061, 0.009
    alpha4, alpha5 = 1.13, 0.74
    join = 0.5 * carrier_fr22
    left = phenomd_eq14_phase_delta_c1(
        torch.tensor([join - 1e-5, join], dtype=torch.float64),
        eta, chi_pn, hm_fr22, hm_flm, hm_fd22, hm_fdlm,
        alpha4, alpha5, base_fr22=carrier_fr22, base_fd22=carrier_fd22,
    )
    assert torch.equal(left, torch.zeros_like(left))

    eps = 1e-9
    x = torch.tensor(join + eps, dtype=torch.float64, requires_grad=True)
    value = phenomd_eq14_phase_delta_c1(
        x, eta, chi_pn, hm_fr22, hm_flm, hm_fd22, hm_fdlm,
        alpha4, alpha5, base_fr22=carrier_fr22, base_fd22=carrier_fd22,
    )
    slope = torch.autograd.grad(value, x)[0]
    assert torch.abs(value) < 1e-12
    assert torch.abs(slope) < 1e-5


def test_eq14_c1_same_mode_identity_survives_distinct_carrier_values():
    x = torch.tensor([0.01, 0.04, 0.09], dtype=torch.float64)
    got = phenomd_eq14_phase_delta_c1(
        x, 0.22, 0.08, 0.083, 0.083, 0.014, 0.014,
        1.13, 0.74, base_fr22=0.061, base_fd22=0.009,
    )
    assert torch.equal(got, torch.zeros_like(got))


def test_assembly_eq14_distinct_carrier_qnm_preserves_dtype_and_autograd():
    x = torch.tensor([0.041, 0.071, 0.109], dtype=torch.float64, requires_grad=True)
    carrier_fr22 = torch.tensor(0.061, dtype=torch.float64, requires_grad=True)
    carrier_fd22 = torch.tensor(0.009, dtype=torch.float64, requires_grad=True)
    kw = _kwargs()
    kw.update(
        phase_qnm_22=0.083,
        phase_qnm_lm=0.119,
        phase_damp_22=0.014,
        phase_damp_lm=0.023,
        base_fr22=carrier_fr22,
        base_fd22=carrier_fd22,
    )
    got = assemble_mode(x, (3, 3), **kw, use_eq14=True)
    assert got.dtype == torch.complex128
    gradients = torch.autograd.grad(
        got.real.sum(), (x, carrier_fr22, carrier_fd22), allow_unused=False,
    )
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert torch.linalg.vector_norm(gradients[0]) > 0
    assert torch.abs(gradients[1]) > 0 and torch.abs(gradients[2]) > 0


def test_join_defaults_are_printed_and_compatibility_pair_is_explicit():
    x = torch.tensor([0.025], dtype=torch.float64)
    kw = _kwargs()
    kw.update(amplitude_22=lambda y: y, use_eq14=False)
    compat = assemble_mode(x, (3, 3), **kw)
    printed = assemble_mode(
        x, (3, 3), **{**kw, "amplitude_join": PRINTED_AMPLITUDE_JOIN,
                       "phase_join": PRINTED_PHASE_JOIN}
    )
    default_kw = {k: v for k, v in kw.items()
                  if k not in ("amplitude_join", "phase_join")}
    default = assemble_mode(x, (3, 3), **default_kw)
    assert torch.allclose(default, printed, atol=1e-14, rtol=1e-14)
    assert not torch.allclose(default.abs(), compat.abs(), atol=1e-14, rtol=1e-14)


def test_mode_21_uses_compatibility_factor_and_public_phase_sign():
    x = torch.tensor([0.001, 0.003], dtype=torch.float64)
    got = assemble_mode(x, (2, 1), **_kwargs(), use_eq14=False)
    assert torch.isfinite(got.real).all() and torch.isfinite(got.imag).all()
    # The clean assembly carries the D public convention h=exp(-i p0).
    carrier = lambda y: 0.7 + 3 * y + .2 * y * y
    expected = -0.5 * carrier(2 * x) - torch.pi / 2
    # The PN factor itself has a small complex tail, so only the public-sign
    # branch is asserted here (its phase perturbation is below 0.1 rad).
    assert torch.all(torch.abs(torch.angle(torch.exp(1j * (torch.angle(got) - expected)))) < 0.1)


def test_projector_is_linear_and_keeps_complex_polarizations():
    z = torch.tensor([1 + 2j, 0.5 - .2j], dtype=torch.complex128)
    harmonic = lambda phi, iota, s, ell, m: complex(1.0 / (ell + abs(m)))
    hp, hc = project_positive_modes({(2, 2): z}, .8, harmonic)
    assert hp.shape == z.shape and hc.shape == z.shape
    z2 = 2 * z
    hp2, hc2 = project_positive_modes({(2, 2): z2}, .8, harmonic)
    assert torch.allclose(hp2, 2 * hp) and torch.allclose(hc2, 2 * hc)


def test_projector_rejects_empty_input_before_reading_modes():
    with pytest.raises(ValueError, match="at least one positive-m mode"):
        project_positive_modes({}, 0.8)


def test_default_projector_is_differentiable_and_vectorized_inclination():
    iota = torch.tensor([0.3, 0.8], dtype=torch.float64, requires_grad=True)
    z = torch.tensor(1.2 - 0.3j, dtype=torch.complex128)
    hp, hc = project_positive_modes({(2, 2): z}, iota)
    assert hp.shape == iota.shape and hc.shape == iota.shape
    grad = torch.autograd.grad((hp.real + hc.imag).sum(), iota)[0]
    assert torch.isfinite(grad).all() and torch.linalg.vector_norm(grad) > 0


def test_custom_tensor_harmonic_keeps_inclination_graph():
    iota = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(1.0 + 0.2j, dtype=torch.complex128)

    def harmonic(phi, theta, spin, ell, m):
        return torch.complex(torch.cos(theta), torch.sin(theta))

    hp, hc = project_positive_modes({(2, 2): z}, iota, harmonic)
    grad = torch.autograd.grad((hp.real + hc.imag).sum(), iota)[0]
    assert torch.isfinite(grad) and torch.abs(grad) > 0


def test_default_projector_inclined_22_matches_analytic_polarizations():
    iota = 0.8
    z = torch.tensor([1.2 - 0.3j], dtype=torch.complex128)
    hp, hc = project_positive_modes({(2, 2): z}, iota)
    c = math.cos(iota)
    n = math.sqrt(5.0 / (64.0 * math.pi))
    assert torch.allclose(hp, z * (n * (1 + c*c)), atol=1e-14, rtol=1e-14)
    assert torch.allclose(hc, z * (-2j * n * c), atol=1e-14, rtol=1e-14)


def test_projector_inclined_22_matches_public_harmonics():
    lal = pytest.importorskip("lal")
    iota = 0.8
    z = torch.tensor([1.2 - 0.3j], dtype=torch.complex128)
    harmonic = lambda phi, theta, s, ell, m: lal.SpinWeightedSphericalHarmonic(float(theta), float(phi), s, ell, m)
    hp, hc = project_positive_modes({(2, 2): z}, iota, harmonic)
    c = math.cos(iota)
    n = math.sqrt(5.0 / (64.0 * math.pi))
    assert torch.allclose(hp, z * (n * (1 + c*c)), atol=1e-14, rtol=1e-14)
    assert torch.allclose(hc, z * (-2j * n * c), atol=1e-14, rtol=1e-14)


def test_projector_has_nonzero_33_at_inclined_angle():
    lal = pytest.importorskip("lal")
    harmonic = lambda phi, theta, s, ell, m: lal.SpinWeightedSphericalHarmonic(float(theta), float(phi), s, ell, m)
    hp, hc = project_positive_modes({(3, 3): torch.ones(5, dtype=torch.complex128)}, .8, harmonic)
    assert torch.linalg.vector_norm(hp) > 0 and torch.linalg.vector_norm(hc) > 0


def test_default_projector_preserves_float32_and_matches_vectorized_high_modes():
    def torch_harmonic(phi, theta, spin, ell, m):
        norm = theta.new_tensor((2 * ell + 1) / (4.0 * math.pi))
        return torch.sqrt(norm) * wigner_d(ell, m, 2, theta)

    inclination = torch.tensor([0.25, 0.8], dtype=torch.float32)
    modes = {
        (3, 3): torch.tensor([1.0 + 0.2j, 0.7 - 0.1j], dtype=torch.complex64),
        (4, 3): torch.tensor([0.4 - 0.5j, -0.2 + 0.8j], dtype=torch.complex64),
    }
    default = project_positive_modes(modes, inclination)
    callback = project_positive_modes(modes, inclination, torch_harmonic)
    assert default[0].dtype == torch.complex64
    assert default[1].dtype == torch.complex64
    assert torch.allclose(default[0], callback[0], atol=2e-6, rtol=2e-6)
    assert torch.allclose(default[1], callback[1], atol=2e-6, rtol=2e-6)


def test_32_amplitude_factor_is_the_requested_leading_term():
    x = torch.tensor([0.02, 0.05], dtype=torch.float64)
    a = assemble_mode(x, (3, 2), **_kwargs(), use_eq14=False)
    kw = _kwargs(); kw["chi_s"] = 0.0
    b = assemble_mode(x, (3, 2), **kw, use_eq14=False)
    # The compatibility recipe intentionally drops the explicit 32 spin term.
    assert torch.allclose(a.abs(), b.abs(), atol=1e-13, rtol=1e-13)


def test_equal_mass_equal_spin_odd_mode_is_zero_and_gauges_are_differentiable():
    kw = _kwargs()
    kw.update(eta=0.25, delta=0.0, chi_s=0.2, chi_a=0.0)
    x = torch.tensor([0.02, 0.05], dtype=torch.float64, requires_grad=True)
    z = assemble_mode(x, (2, 1), **kw, use_eq14=False)
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-14, rtol=0)
    # Tensor-valued extrinsic gauges remain in the autograd graph.
    ref = torch.tensor(0.13, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(0.27, dtype=torch.float64, requires_grad=True)
    kw.update(reference_gauge=ref, phic=phi, delta=0.2)
    zz = assemble_mode(x, (2, 2), **kw, use_eq14=False)
    torch.autograd.grad(zz.real.sum(), (ref, phi), allow_unused=False)
