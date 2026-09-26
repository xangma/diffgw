"""Production XP routing contracts for the clean native MSA v223 adapter."""

import math

import pytest
import torch

import torchwave.approximants.imrphenom_xp as xp_module
from torchwave.approximants.imrphenom_xp import IMRPhenomXP
from torchwave.approximants.precession_msa_waveform_clean import waveform_v223


DTYPE = torch.float64


def _case():
    return dict(
        mass1=35.0,
        mass2=25.0,
        spin1x=0.20,
        spin1y=0.10,
        spin1z=0.30,
        spin2x=-0.10,
        spin2y=0.05,
        spin2z=-0.20,
        distance=150.0,
        inclination=0.70,
        phic=0.30,
        tc=0.0,
        f_ref=20.0,
    )


@pytest.mark.parametrize("convention", (0, 1))
def test_supported_tuple_agrees_with_clean_helper(convention):
    f = torch.linspace(20.0, 220.0, 17, dtype=DTYPE)
    params = _case()
    model = IMRPhenomXP()

    got = model(f, **params, prec_version=223, convention=convention,
                final_spin_mod=3)
    want = waveform_v223(
        f, **params, convention=convention, final_spin_mod=3,
        xas_model=model.xas_model,
    )

    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_omitted_choices_use_msa_v223_and_match_explicit_contract():
    f = torch.linspace(20.0, 220.0, 17, dtype=DTYPE)
    params = {**_case(), "f_ref": None}
    model = IMRPhenomXP()

    default = model(f, **params)
    explicit = model(
        f, **params, prec_version=223, convention=1, final_spin_mod=3,
    )
    for actual, expected in zip(default, explicit):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_explicit_nnlo_and_zero_final_spin_keep_existing_path(monkeypatch):
    f = torch.tensor([20.0, 60.0, 120.0], dtype=DTYPE)
    params = _case()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("clean v223 helper was called for an NNLO tuple")

    monkeypatch.setattr(xp_module, "waveform_v223", fail_if_called)
    model = IMRPhenomXP()
    hp_nnlo, hc_nnlo = model(
        f, **params, precession_method="nnlo", final_spin_mod=0,
    )
    hp_msa_zero, hc_msa_zero = model(
        f, **params, prec_version=223, final_spin_mod=0,
    )
    assert torch.isfinite(hp_nnlo).all() and torch.isfinite(hc_nnlo).all()
    assert torch.isfinite(hp_msa_zero).all() and torch.isfinite(hc_msa_zero).all()


def test_broadcast_rows_match_scalar_calls():
    f = torch.linspace(20.0, 200.0, 13, dtype=DTYPE)
    batch = dict(
        mass1=torch.tensor([35.0, 20.0, 42.0], dtype=DTYPE),
        mass2=torch.tensor([25.0, 30.0, 17.0], dtype=DTYPE),
        spin1x=torch.tensor([0.20, -0.10, 0.30], dtype=DTYPE),
        spin1y=torch.tensor([0.10, 0.05, -0.15], dtype=DTYPE),
        spin1z=torch.tensor([0.30, -0.20, -0.30], dtype=DTYPE),
        spin2x=torch.tensor([-0.10, 0.20, 0.12], dtype=DTYPE),
        spin2y=torch.tensor([0.05, 0.00, 0.08], dtype=DTYPE),
        spin2z=torch.tensor([-0.20, 0.10, 0.20], dtype=DTYPE),
        distance=torch.tensor([150.0, 200.0, 500.0], dtype=DTYPE),
        inclination=torch.tensor([0.70, 0.40, 1.10], dtype=DTYPE),
        phic=torch.tensor([0.30, 0.00, 0.71], dtype=DTYPE),
        tc=torch.tensor([0.0, 0.002, -0.001], dtype=DTYPE),
        f_ref=torch.tensor([20.0, 30.0, 35.0], dtype=DTYPE),
    )
    model = IMRPhenomXP()
    hp_batch, hc_batch = model(
        f, **batch, prec_version=223, convention=0, final_spin_mod=3,
    )
    assert hp_batch.shape == (3, f.numel())
    assert hc_batch.shape == (3, f.numel())
    for row in range(3):
        scalar = {name: value[row] for name, value in batch.items()}
        hp_scalar, hc_scalar = model(
            f, **scalar, prec_version=223, convention=0, final_spin_mod=3,
        )
        torch.testing.assert_close(hp_batch[row], hp_scalar, rtol=1e-15, atol=1e-25)
        torch.testing.assert_close(hc_batch[row], hc_scalar, rtol=1e-15, atol=1e-25)


def test_exact_aligned_scalar_and_singleton_use_the_same_carrier(monkeypatch):
    f = torch.tensor([25.0, 40.0, 80.0], dtype=DTYPE)
    params = dict(
        mass1=100.0, mass2=5.0, spin1z=-0.95, spin2z=0.0,
        distance=150.0, inclination=0.7, phic=0.3, f_ref=40.0,
    )
    calls = []
    original = xp_module.waveform_v223

    def record_helper(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(xp_module, "waveform_v223", record_helper)
    model = IMRPhenomXP()
    hp_scalar, hc_scalar = model(
        f, **params, prec_version=223, convention=1, final_spin_mod=3,
    )
    singleton_params = {
        name: torch.tensor([value], dtype=DTYPE)
        for name, value in params.items()
    }
    hp_singleton, hc_singleton = model(
        f, **singleton_params, prec_version=223, convention=1,
        final_spin_mod=3,
    )
    assert not calls
    assert hp_singleton.shape == (1, f.numel())
    assert hc_singleton.shape == (1, f.numel())
    torch.testing.assert_close(hp_singleton[0], hp_scalar, rtol=0.0, atol=0.0)
    torch.testing.assert_close(hc_singleton[0], hc_scalar, rtol=0.0, atol=0.0)
    assert torch.isfinite(hp_singleton).all() and torch.isfinite(hc_singleton).all()


def test_mixed_exact_aligned_and_clean_rows_dispatch_independently(monkeypatch):
    f = torch.linspace(20.0, 160.0, 9, dtype=DTYPE)
    exact = dict(
        mass1=100.0, mass2=5.0, spin1x=0.0, spin1y=0.0,
        spin1z=-0.95, spin2x=0.0, spin2y=0.0, spin2z=0.0,
        distance=150.0, inclination=0.7, phic=0.3, tc=0.0, f_ref=40.0,
    )
    clean = _case()
    calls = []
    original = xp_module.waveform_v223

    def record_helper(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(xp_module, "waveform_v223", record_helper)
    batch = {
        name: torch.tensor([exact[name], clean[name]], dtype=DTYPE)
        for name in exact
    }
    hp_batch, hc_batch = IMRPhenomXP()(
        f, **batch, prec_version=223, convention=1, final_spin_mod=3,
    )
    # Only the genuinely precessing row uses the clean helper; the exact
    # aligned row is evaluated through the carrier limit independently.
    assert len(calls) == 1
    for row, scalar_params in enumerate((exact, clean)):
        hp_scalar, hc_scalar = IMRPhenomXP()(
            f, **scalar_params, prec_version=223, convention=1,
            final_spin_mod=3,
        )
        torch.testing.assert_close(hp_batch[row], hp_scalar, rtol=0.0, atol=0.0)
        torch.testing.assert_close(hc_batch[row], hc_scalar, rtol=0.0, atol=0.0)


def test_mixed_clean_and_invalid_l_plus_s_rows_keep_valid_row_on_clean_route(monkeypatch):
    f = torch.linspace(25.0, 80.0, 4, dtype=DTYPE)
    valid = _case()
    invalid = {
        **valid,
        "mass1": 100.0, "mass2": 5.0,
        "spin1x": 1e-9, "spin1y": 0.0, "spin1z": -0.95,
        "spin2x": 0.0, "spin2y": 0.0, "spin2z": 0.0,
        "f_ref": 40.0,
    }
    calls = []
    original = xp_module.waveform_v223

    def record_helper(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(xp_module, "waveform_v223", record_helper)
    batch = {
        name: torch.tensor([valid[name], invalid[name]], dtype=DTYPE)
        for name in valid
    }
    with pytest.raises(ValueError, match=r"L \+ S_parallel <= 0"):
        IMRPhenomXP()(
            f, **batch, prec_version=223, convention=1, final_spin_mod=3,
        )
    # The valid row is evaluated before the invalid row raises; a batch-wide
    # legacy fallback would leave this count at zero.
    assert len(calls) == 1


def test_tiny_nonzero_in_plane_spin_does_not_use_exact_aligned_carrier(monkeypatch):
    f = torch.tensor([25.0, 40.0, 80.0], dtype=DTYPE)
    params = dict(
        mass1=100.0, mass2=5.0, spin1x=1e-9, spin1y=0.0,
        spin1z=-0.95, spin2x=0.0, spin2y=0.0, spin2z=0.0,
        distance=150.0, inclination=0.7, phic=0.3, f_ref=40.0,
    )
    calls = []
    original = xp_module.waveform_v223

    def record_helper(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(xp_module, "waveform_v223", record_helper)
    with pytest.raises(ValueError, match=r"L \+ S_parallel <= 0"):
        IMRPhenomXP()(
            f, **params, prec_version=223, convention=1, final_spin_mod=3,
        )
    assert not calls


def test_frequency_shape_and_omitted_carrier_reference_are_preserved():
    f = torch.tensor([[20.0, 40.0], [80.0, 160.0]], dtype=DTYPE)
    params = _case()
    model = IMRPhenomXP()
    hp, hc = model(
        f, **{**params, "f_ref": None}, prec_version=223,
        convention=1, final_spin_mod=3,
    )
    hp_ref, hc_ref = model(
        f, **{**params, "f_ref": 20.0}, prec_version=223,
        convention=1, final_spin_mod=3,
    )
    assert hp.shape == f.shape
    assert hc.shape == f.shape
    # The angle chart uses 20 Hz when the public reference is omitted, but
    # XAS has a distinct no-reference carrier phase. Preserve that semantic
    # distinction through the XP composition.
    assert torch.linalg.vector_norm(hp - hp_ref) / torch.linalg.vector_norm(hp_ref) > 0.01
    assert torch.linalg.vector_norm(hc - hc_ref) / torch.linalg.vector_norm(hc_ref) > 0.01


def test_distance_and_time_shift_contracts():
    f = torch.tensor([0.0, 20.0, 40.0, 100.0], dtype=DTYPE)
    params = _case()
    model = IMRPhenomXP()
    hp_0, hc_0 = model(f, **params, prec_version=223, convention=0,
                       final_spin_mod=3)
    hp_d, hc_d = model(f, **{**params, "distance": 2.0 * params["distance"]},
                       prec_version=223, convention=0, final_spin_mod=3)
    torch.testing.assert_close(hp_d, hp_0 / 2.0, rtol=2e-14, atol=1e-35)
    torch.testing.assert_close(hc_d, hc_0 / 2.0, rtol=2e-14, atol=1e-35)

    dt = 0.013
    hp_t, hc_t = model(f, **{**params, "tc": dt}, prec_version=223,
                       convention=0, final_spin_mod=3)
    shift = torch.exp(-2j * math.pi * f * dt)
    scale = max(hp_0.abs().max().item(), hc_0.abs().max().item())
    assert torch.max(torch.abs(hp_t - hp_0 * shift)).item() / scale < 1e-12
    assert torch.max(torch.abs(hc_t - hc_0 * shift)).item() / scale < 1e-12
    assert hp_t[0] == 0 and hc_t[0] == 0


def test_gradients_through_full_three_component_spins_are_finite():
    f = torch.linspace(25.0, 200.0, 15, dtype=DTYPE)
    values = [torch.tensor(value, dtype=DTYPE, requires_grad=True) for value in
              (35.0, 20.0, 0.20, 0.15, 0.10, 0.10, 0.05, -0.10)]
    names = ("mass1", "mass2", "spin1x", "spin1y", "spin1z",
             "spin2x", "spin2y", "spin2z")
    params = dict(zip(names, values))
    hp, hc = IMRPhenomXP()(
        f, **params, inclination=0.70, phic=0.30, f_ref=20.0,
        prec_version=223, convention=0, final_spin_mod=3,
    )
    (hp.abs().square() + hc.abs().square()).sum().backward()
    for value in values:
        assert value.grad is not None
        assert torch.isfinite(value.grad)
        assert value.grad.abs() > 0.0


@pytest.mark.parametrize(
    "params",
    (
        dict(mass1=25.0, mass2=25.0, spin1x=0.20, spin1y=0.10,
             spin1z=0.0, spin2x=0.0, spin2y=0.0, spin2z=0.0),
        dict(mass1=25.000001, mass2=25.0, spin1x=0.20, spin1y=0.10,
             spin1z=0.0, spin2x=0.0, spin2y=0.0, spin2z=0.0),
        dict(mass1=35.0, mass2=25.0, spin1x=0.0, spin1y=0.0,
             spin1z=0.30, spin2x=0.0, spin2y=0.0, spin2z=-0.20),
    ),
)
def test_equal_aligned_and_boundary_rows_are_finite(params):
    f = torch.linspace(20.0, 200.0, 11, dtype=DTYPE)
    hp, hc = IMRPhenomXP()(
        f, **params, distance=100.0, inclination=0.7, phic=0.2,
        f_ref=20.0, prec_version=223, convention=0, final_spin_mod=3,
    )
    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()


def test_unsupported_versions_do_not_route_to_clean_helper(monkeypatch):
    f = torch.tensor([20.0, 60.0], dtype=DTYPE)
    params = _case()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("clean v223 helper was called for an unsupported tuple")

    monkeypatch.setattr(xp_module, "waveform_v223", fail_if_called)
    hp, hc = IMRPhenomXP()(
        f, **params, prec_version=222, convention=0, final_spin_mod=3,
    )
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
    with pytest.raises(ValueError, match="precession version"):
        IMRPhenomXP()(f, **params, prec_version=999, convention=0,
                      final_spin_mod=3)


def test_nonzero_phi0_uses_clean_source_chart(monkeypatch):
    f = torch.linspace(20.0, 160.0, 9, dtype=DTYPE)
    params = _case()
    calls = []
    original = xp_module.waveform_v223

    def record_helper(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(xp_module, "waveform_v223", record_helper)
    model = IMRPhenomXP()
    model(f, **params, phi0=0.0, prec_version=223, convention=1,
          final_spin_mod=3)
    hp_a, hc_a = model(
        f, **params, phi0=0.4, prec_version=223, convention=1,
        final_spin_mod=3,
    )
    hp_b, hc_b = model(
        f, **params, phi0=0.7, prec_version=223, convention=1,
        final_spin_mod=3,
    )
    assert len(calls) == 3
    assert torch.isfinite(hp_a).all() and torch.isfinite(hc_a).all()
    assert torch.isfinite(hp_b).all() and torch.isfinite(hc_b).all()
    assert torch.max(torch.abs(hp_a - hp_b)).item() > 0.0
    assert torch.max(torch.abs(hc_a - hc_b)).item() > 0.0


def test_mixed_phi0_rows_match_independent_scalar_routes():
    f = torch.linspace(20.0, 160.0, 9, dtype=DTYPE)
    params = _case()
    model = IMRPhenomXP()
    phi0 = torch.tensor([0.0, 0.4], dtype=DTYPE)
    batched = model(
        f, **params, phi0=phi0,
        prec_version=223, convention=1, final_spin_mod=3,
    )
    for row in range(2):
        scalar = model(
            f, **params, phi0=phi0[row],
            prec_version=223, convention=1, final_spin_mod=3,
        )
        for actual, expected in zip(batched, scalar):
            # Batched and scalar chart arithmetic can round one ulp apart.
            torch.testing.assert_close(actual[row], expected, rtol=1e-15, atol=0.0)


def test_one_spin_physical_row_survives_generic_azimuth_batch():
    f = torch.linspace(20.0, 160.0, 9, dtype=DTYPE)
    params = {
        **_case(),
        "spin2x": torch.tensor([0.0, -0.2], dtype=DTYPE),
        "spin2y": torch.tensor([0.0, 0.1], dtype=DTYPE),
    }
    model = IMRPhenomXP()
    batched = model(
        f, **params, phi0=0.4,
        prec_version=223, convention=1, final_spin_mod=3,
    )
    for row in range(2):
        scalar_params = {
            **params, "spin2x": params["spin2x"][row],
            "spin2y": params["spin2y"][row],
        }
        scalar = model(
            f, **scalar_params, phi0=0.4,
            prec_version=223, convention=1, final_spin_mod=3,
        )
        for actual, expected in zip(batched, scalar):
            # The mixed clean branches can differ by one rounding step.
            torch.testing.assert_close(actual[row], expected, rtol=1e-15, atol=0.0)


def test_aligned_nonzero_phi0_has_expected_m2_phase_rotation():
    f = torch.linspace(20.0, 160.0, 9, dtype=DTYPE)
    params = {
        **_case(),
        "spin1x": 0.0,
        "spin1y": 0.0,
        "spin2x": 0.0,
        "spin2y": 0.0,
    }
    model = IMRPhenomXP()
    phi_a, phi_b = 0.4, 0.7
    hp_a, hc_a = model(
        f, **params, phi0=phi_a, prec_version=223, convention=1,
        final_spin_mod=3,
    )
    hp_b, hc_b = model(
        f, **params, phi0=phi_b, prec_version=223, convention=1,
        final_spin_mod=3,
    )
    phase = torch.exp(torch.tensor(-2j * (phi_b - phi_a), dtype=torch.complex128))
    torch.testing.assert_close(hp_b, hp_a * phase, rtol=2e-14, atol=1e-35)
    torch.testing.assert_close(hc_b, hc_a * phase, rtol=2e-14, atol=1e-35)


def test_final_spin_version_alias_and_conflict_are_explicit():
    f = torch.tensor([20.0, 60.0], dtype=DTYPE)
    params = _case()
    model = IMRPhenomXP()
    direct = model(f, **params, prec_version=223, convention=0,
                   final_spin_mod=3)
    alias = model(f, **params, prec_version=223, convention=0,
                  final_spin_version=3)
    for actual, expected in zip(alias, direct):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="must agree"):
        model(f, **params, prec_version=223, convention=0,
              final_spin_mod=0, final_spin_version=3)
