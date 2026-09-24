import math

import pytest
import torch

from torchwave.approximants.precession_msa_waveform_clean import (
    compose_msa_waveform,
    finalspin_3,
    msa_final_spin,
    msa_final_spin_normalization_report,
    msa_final_spin_squared,
    transport_msa_angles_for_phi_ref,
    waveform,
)
from torchwave.approximants.precession_twisting_clean import (
    project_twisted_positive_modes,
    twist_mode,
)


class _AnalyticCarrier:
    """Batch-capable carrier double that records composition boundary inputs."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        f = kwargs["f"]
        a_fin = kwargs["_final_spin_override"]
        scale = 1.0 + 0.07 * a_fin
        carrier_phase = torch.exp(1j * 0.31 * kwargs["phi_c"])
        if scale.ndim:
            scale = scale.unsqueeze(-1)
        if carrier_phase.ndim:
            carrier_phase = carrier_phase.unsqueeze(-1)
        hp = carrier_phase * scale * (1.0 + 0.2j) * torch.pow(f, -0.5)
        return hp, torch.zeros_like(hp)


def test_msa_final_spin_aligned_limit_exposes_normalization_gap():
    a_parallel = torch.tensor([0.7, -0.4], dtype=torch.float64)
    remnant_fraction = torch.tensor([0.95, 0.9], dtype=torch.float64)
    s_parallel = torch.tensor([0.18, -0.11], dtype=torch.float64)
    mean_s1_l = torch.tensor([0.12, -0.03], dtype=torch.float64)
    mean_s2_l = s_parallel - mean_s1_l
    sav = s_parallel.abs()

    remnant = msa_final_spin(
        sav, mean_s1_l, mean_s2_l, a_parallel, remnant_fraction,
        normalization="remnant",
    )
    printed = finalspin_3(
        sav, mean_s1_l, mean_s2_l, a_parallel, remnant_fraction,
        normalization="paper",
    )
    report = msa_final_spin_normalization_report(a_parallel, remnant_fraction)

    # Eq. (afin_L_fin) gives L_fin = M_fin^2 a_parallel - S_parallel.
    # Consequently the remnant-normalized aligned limit is a_parallel, while
    # the denominator printed in Eq. (finalspin_3) is the source mass.
    assert torch.allclose(remnant, a_parallel, atol=2e-13, rtol=0.0)
    assert torch.allclose(printed, remnant_fraction.square() * a_parallel, atol=2e-13, rtol=0.0)
    assert torch.allclose(report["aligned_remnant_normalized"], a_parallel)
    assert torch.allclose(report["aligned_total_mass_normalized"], printed)
    assert torch.allclose(report["total_minus_remnant"], printed - a_parallel)


def test_msa_final_spin_compatibility_separates_initial_and_mean_projections():
    sav = torch.tensor([0.27, 0.34], dtype=torch.float64)
    mean_s1_l = torch.tensor([0.07, -0.03], dtype=torch.float64)
    mean_s2_l = torch.tensor([0.02, 0.11], dtype=torch.float64)
    initial_s1_l = torch.tensor([0.12, -0.01], dtype=torch.float64)
    initial_s2_l = torch.tensor([-0.04, 0.08], dtype=torch.float64)
    a_parallel = torch.tensor([0.64, 0.52], dtype=torch.float64)
    mf = torch.tensor([0.94, 0.91], dtype=torch.float64)

    # The default remnant mode preserves the original helper contract when
    # initial projections equal the means.  With distinct inputs, changing
    # only the initial pair changes L_fin, while changing only the mean pair
    # changes the averaged cross term.
    legacy = msa_final_spin_squared(sav, mean_s1_l, mean_s2_l, a_parallel, mf)
    equal_projection = msa_final_spin_squared(
        sav, mean_s1_l, mean_s2_l, a_parallel, mf,
        initial_s1_l=mean_s1_l, initial_s2_l=mean_s2_l,
    )
    compatible = msa_final_spin_squared(
        sav, mean_s1_l, mean_s2_l, a_parallel, mf,
        initial_s1_l=initial_s1_l, initial_s2_l=initial_s2_l,
        l_fin_normalization="initial",
    )
    initial_changed = msa_final_spin_squared(
        sav, mean_s1_l, mean_s2_l, a_parallel, mf,
        initial_s1_l=initial_s1_l + 0.01, initial_s2_l=initial_s2_l,
        l_fin_normalization="initial",
    )
    mean_changed = msa_final_spin_squared(
        sav, mean_s1_l + 0.01, mean_s2_l, a_parallel, mf,
        initial_s1_l=initial_s1_l, initial_s2_l=initial_s2_l,
        l_fin_normalization="initial",
    )

    assert torch.allclose(equal_projection, legacy, atol=1e-14, rtol=0.0)
    assert not torch.allclose(compatible, legacy, atol=1e-10, rtol=0.0)
    assert not torch.allclose(initial_changed, compatible, atol=1e-10, rtol=0.0)
    assert not torch.allclose(mean_changed, compatible, atol=1e-10, rtol=0.0)

    # In the compatibility mode the aligned fit is already an initial-M^2
    # L_fin scalar, so changing only the auxiliary remnant fraction has no
    # effect on the squared MSA magnitude.
    changed_mf = msa_final_spin_squared(
        sav, mean_s1_l, mean_s2_l, a_parallel, mf * 0.97,
        initial_s1_l=initial_s1_l, initial_s2_l=initial_s2_l,
        l_fin_normalization="initial",
    )
    assert torch.allclose(changed_mf, compatible, atol=1e-14, rtol=0.0)
    with pytest.raises(ValueError, match="l_fin_normalization"):
        msa_final_spin_squared(
            sav, mean_s1_l, mean_s2_l, a_parallel, mf,
            l_fin_normalization="unknown",
        )


def test_msa_final_spin_formula_broadcasts_and_has_gradients():
    sav = torch.tensor([[0.23], [0.31]], dtype=torch.float64, requires_grad=True)
    s1_l = torch.tensor([[0.08, 0.02, -0.04]], dtype=torch.float64, requires_grad=True)
    s2_l = torch.tensor([[-0.01, 0.05, 0.07]], dtype=torch.float64, requires_grad=True)
    a_parallel = torch.tensor([0.63, -0.48, 0.2], dtype=torch.float64, requires_grad=True)
    mf = torch.tensor([0.93, 0.95, 0.97], dtype=torch.float64, requires_grad=True)

    squared = msa_final_spin_squared(sav, s1_l, s2_l, a_parallel, mf)
    assert squared.shape == (2, 3)
    assert torch.isfinite(squared).all()

    # Check the public squared-input path against the magnitude path without
    # duplicating the implementation's intermediate arithmetic here.
    squared_input = msa_final_spin_squared(
        sav.square(), s1_l, s2_l, a_parallel, mf, average_is_squared=True,
    )
    assert torch.allclose(squared, squared_input, atol=1e-13, rtol=1e-13)

    loss = msa_final_spin(
        sav, s1_l, s2_l, a_parallel, mf, normalization="remnant",
    ).square().sum()
    grads = torch.autograd.grad(loss, (sav, s1_l, s2_l, a_parallel, mf))
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert all(torch.any(grad.abs() > 0.0) for grad in grads)


def test_msa_final_spin_reuses_input_validation():
    with pytest.raises(ValueError, match="remnant_mass_fraction"):
        msa_final_spin(0.2, 0.08, -0.03, 0.6, 0.0)
    with pytest.raises(ValueError, match="aligned_kerr_parameter"):
        msa_final_spin(0.2, 0.08, -0.03, torch.tensor(float("nan")), 0.9)
    with pytest.raises(ValueError, match="supplied together"):
        msa_final_spin(0.2, 0.08, -0.03, 0.6, 0.9, initial_s1_l=0.04)


def test_composition_injects_angles_forwards_override_and_supports_dc():
    f = torch.tensor([0.0, 20.0, 40.0, 80.0], dtype=torch.float64)
    alpha = torch.tensor([0.1, 0.14, 0.2, 0.31], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor([0.2, 0.21, 0.23, 0.28], dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor([0.0, -0.1, -0.16, -0.23], dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(0.65, dtype=torch.float64, requires_grad=True)
    zeta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    phi_j = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    final_spin = torch.tensor(0.71, dtype=torch.float64, requires_grad=True)
    carrier = _AnalyticCarrier()

    hp, hc = compose_msa_waveform(
        f, 35.0, 25.0, 0.2, -0.1,
        {"alpha": alpha, "beta": beta, "gamma": gamma},
        {"alpha0": 0.4, "theta_jn": theta, "zeta": zeta, "phi_j": phi_j},
        final_spin,
        distance=500.0, f_ref=20.0, convention=0, xas_model=carrier,
    )
    assert hp.shape == f.shape and hc.shape == f.shape
    assert hp.dtype == torch.complex128 and hc.dtype == torch.complex128
    assert hp[0] == 0 and hc[0] == 0
    assert torch.isfinite(hp[1:]).all() and torch.isfinite(hc[1:]).all()
    assert len(carrier.calls) == 1
    assert torch.equal(carrier.calls[0]["_final_spin_override"], final_spin)
    assert torch.equal(carrier.calls[0]["f"], torch.tensor([20., 20., 40., 80.], dtype=torch.float64))

    # A phase-sensitive scalar keeps the carrier and polarization geometry in
    # the autograd graph; a magnitude-only loss would erase those rotations.
    loss = hp[1:].real.sum() + 0.4 * hc[1:].imag.sum()
    grads = torch.autograd.grad(loss, (alpha, beta, gamma, theta, zeta, phi_j, final_spin))
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert all(torch.any(grad.abs() > 0.0) for grad in grads)


def test_absolute_public_angles_equal_relative_angle_injection():
    f = torch.tensor([20.0, 40.0, 100.0], dtype=torch.float64)
    relative = {
        "alpha": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        "beta": torch.tensor(0.22, dtype=torch.float64),
        "gamma": torch.tensor([0.0, -0.1, -0.2], dtype=torch.float64),
    }
    absolute = {"alpha": relative["alpha"] + 0.4, "beta": relative["beta"], "gamma": relative["gamma"]}
    geometry = (0.4, 0.65, 0.7, 0.25)

    first = compose_msa_waveform(
        f, 35.0, 25.0, 0.2, -0.1, relative, geometry, 0.71,
        xas_model=_AnalyticCarrier(), convention=0,
    )
    second = compose_msa_waveform(
        f, 35.0, 25.0, 0.2, -0.1, absolute, geometry, 0.71,
        xas_model=_AnalyticCarrier(), convention=0, angles_are_absolute=True,
    )
    assert torch.allclose(first[0], second[0], atol=1e-30, rtol=2e-13)
    assert torch.allclose(first[1], second[1], atol=1e-30, rtol=2e-13)


def test_composition_and_native_wrapper_broadcast_parameter_rows():
    f = torch.tensor([20.0, 40.0, 80.0], dtype=torch.float64)
    carrier = _AnalyticCarrier()
    hp, hc = compose_msa_waveform(
        f,
        torch.tensor([35.0, 45.0]), torch.tensor([25.0, 15.0]),
        torch.tensor([0.2, 0.3]), torch.tensor([-0.1, 0.2]),
        {"alpha": torch.tensor([0.1, 0.2, 0.3]), "beta": 0.2, "gamma": 0.0},
        {"alpha0": 0.4, "theta_jn": torch.tensor([0.6, 0.7]), "zeta": 0.5, "phi_j": 0.2},
        torch.tensor([0.7, 0.65]), xas_model=carrier,
    )
    assert hp.shape == (2, 3) and hc.shape == (2, 3)
    assert len(carrier.calls) == 1
    assert carrier.calls[0]["_final_spin_override"].shape == (2,)
    assert carrier.calls[0]["f"].shape == f.shape
    for row, (m1, m2, chi1, chi2, theta, final_spin) in enumerate((
        (35.0, 25.0, 0.2, -0.1, 0.6, 0.7),
        (45.0, 15.0, 0.3, 0.2, 0.7, 0.65),
    )):
        # Match the original float32 fixture inputs before broadcasting to f.
        row_hp, row_hc = compose_msa_waveform(
            f, *(torch.tensor(value) for value in (m1, m2, chi1, chi2)),
            {"alpha": torch.tensor([0.1, 0.2, 0.3]), "beta": 0.2, "gamma": 0.0},
            {"alpha0": 0.4, "theta_jn": torch.tensor(theta), "zeta": 0.5, "phi_j": 0.2},
            torch.tensor(final_spin), xas_model=_AnalyticCarrier(),
        )
        torch.testing.assert_close(hp[row], row_hp, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(hc[row], row_hc, rtol=1e-12, atol=1e-12)

    native = waveform(
        f, 35.0, 25.0, 0.2, 0.1, 0.2, -0.1, 0.15, -0.1,
        distance=500.0, inclination=0.7, phic=0.0, f_ref=20.0,
        convention=0, final_spin_mod=3,
    )
    assert native[0].shape == f.shape and native[1].shape == f.shape
    mixed = waveform(
        f,
        torch.tensor([35.0, 36.0]), torch.tensor([25.0, 24.0]),
        spin1x=torch.tensor([0.2, 0.21]), spin1y=0.1, spin1z=0.2,
        spin2x=-0.1, spin2y=torch.tensor([0.15, 0.12]), spin2z=-0.1,
        distance=500.0, inclination=0.7, phic=0.0, f_ref=20.0,
        convention=0, final_spin_mod=3,
    )
    assert mixed[0].shape == (2, f.numel()) and mixed[1].shape == (2, f.numel())
    with pytest.raises(TypeError, match="prec_version"):
        waveform(f, 35.0, 25.0, prec_version=223)


def test_angle_frequency_rows_broadcast_against_scalar_physical_parameters():
    f = torch.tensor([20.0, 40.0, 80.0], dtype=torch.float64)
    angles = {
        "alpha": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64),
        "beta": torch.tensor([[0.2, 0.21, 0.22], [0.23, 0.24, 0.25]], dtype=torch.float64),
        "gamma": torch.tensor([[0.0, -0.1, -0.2], [0.1, 0.0, -0.1]], dtype=torch.float64),
    }
    carrier = _AnalyticCarrier()
    hp, hc = compose_msa_waveform(
        f, 35.0, 25.0, 0.2, -0.1, angles,
        (0.4, 0.65, 0.7, 0.25), 0.71,
        xas_model=carrier, angles_are_absolute=True,
    )
    assert hp.shape == (2, f.numel()) and hc.shape == (2, f.numel())
    assert len(carrier.calls) == 1
    assert carrier.calls[0]["_final_spin_override"].shape == (2,)
    for row in range(2):
        row_hp, row_hc = compose_msa_waveform(
            f, 35.0, 25.0, 0.2, -0.1,
            {name: value[row] for name, value in angles.items()},
            (0.4, 0.65, 0.7, 0.25), 0.71,
            xas_model=_AnalyticCarrier(), angles_are_absolute=True,
        )
        torch.testing.assert_close(hp[row], row_hp, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(hc[row], row_hc, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()


def test_published_rotation_identities_hold_independently_of_composition():
    alpha = torch.tensor([0.13, 0.41, 0.92], dtype=torch.float64)
    gamma = torch.tensor([-0.2, 0.07, -0.31], dtype=torch.float64)
    beta = torch.tensor([0.24, 0.29, 0.33], dtype=torch.float64)
    delta = 0.37
    h = torch.tensor([1.0 + 0.2j, 0.4 - 0.7j, -0.2 + 0.9j], dtype=torch.complex128)

    base = twist_mode(2, 2, alpha, gamma, beta, {2: h})
    shifted = twist_mode(2, 2, alpha + delta, gamma, beta, {2: h})
    alpha_factor = torch.exp(torch.tensor(-2j * delta, dtype=torch.complex128))
    assert torch.allclose(shifted, alpha_factor * base, atol=2e-15, rtol=0.0)

    modes = {(2, 2): h}
    projected = project_twisted_positive_modes(modes, alpha, gamma, beta, 0.61, 0.27)
    projected_half_turn = project_twisted_positive_modes(
        modes, alpha, gamma, beta, 0.61, 0.27 + math.pi / 2.0,
    )
    assert torch.allclose(projected_half_turn[0], -projected[0], atol=2e-15, rtol=0.0)
    assert torch.allclose(projected_half_turn[1], -projected[1], atol=2e-15, rtol=0.0)


@pytest.mark.parametrize("convention", (0, 1))
def test_source_phase_transport_matches_current_geometry_for_both_conventions(convention):
    """Appendix C transport preserves the quadrupole waveform at nonzero phiRef."""
    f = torch.tensor([20.0, 40.0, 100.0], dtype=torch.float64)
    angles = {
        "alpha": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        "gamma": torch.tensor([0.0, -0.1, -0.2], dtype=torch.float64),
        "cos_beta": torch.cos(torch.tensor(0.22, dtype=torch.float64)),
    }
    geometry_zero = {"alpha0": 0.4, "theta_jn": 0.65, "zeta": 0.7, "phi_j": 0.25}
    geometry_phi = {"alpha0": 0.67, "theta_jn": 0.67, "zeta": 0.81, "phi_j": -0.11}
    shifted, anchor, details = transport_msa_angles_for_phi_ref(
        angles, geometry_zero, geometry_phi, convention=convention,
    )
    assert torch.allclose(details["delta_alpha"], torch.tensor(0.27, dtype=torch.float64), atol=1e-14, rtol=0.0)
    assert torch.allclose(details["delta_phi_j"], torch.tensor(-0.36, dtype=torch.float64), atol=1e-14, rtol=0.0)
    expected_gamma = 0.36 if convention == 1 else 0.0
    assert torch.allclose(details["delta_gamma"], torch.tensor(expected_gamma, dtype=torch.float64), atol=1e-14, rtol=0.0)
    direct = compose_msa_waveform(
        f, 35.0, 25.0, 0.2, -0.1, angles, geometry_phi, 0.71,
        xas_model=_AnalyticCarrier(), convention=convention,
    )
    transported = compose_msa_waveform(
        f, 35.0, 25.0, 0.2, -0.1, shifted, anchor, 0.71,
        xas_model=_AnalyticCarrier(), convention=convention,
    )
    assert torch.allclose(direct[0], transported[0], atol=3e-15, rtol=0.0)
    assert torch.allclose(direct[1], transported[1], atol=3e-15, rtol=0.0)


def test_source_phase_transport_aligned_branch_keeps_gauge_fallback():
    geometry_zero = {"alpha0": 0.0, "theta_jn": 0.7, "zeta": 0.0, "phi_j": math.pi / 2.0}
    geometry_phi = {"alpha0": 0.0, "theta_jn": 0.7, "zeta": 0.0, "phi_j": math.pi / 2.0 - 0.2}
    angles = {"alpha": torch.tensor([0.0, 0.1]), "gamma": torch.tensor([0.0, -0.1]), "beta": torch.zeros(2)}
    shifted, anchor, details = transport_msa_angles_for_phi_ref(
        angles, geometry_zero, geometry_phi, convention=1, aligned=True,
    )
    assert bool(details["aligned_gamma_transport_omitted"])
    assert torch.allclose(shifted["gamma"], angles["gamma"].to(torch.float64), atol=0.0, rtol=0.0)
    assert torch.allclose(anchor["phi_j"], torch.tensor(math.pi / 2.0 - 0.2, dtype=torch.float64), atol=1e-15, rtol=0.0)
