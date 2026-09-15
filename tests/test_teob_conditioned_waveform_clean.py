"""Contract tests for the opt-in clean TEOB TD-to-FD assembly seam."""

from types import SimpleNamespace

import pytest
import torch

from torchwave.approximants.eob import teob_conditioned_waveform_clean as adapter
from torchwave.approximants.eob.td_fd_conditioning_clean import (
    ConditionedFD,
    fft_length_and_sampling,
    plan_td_support,
)


def _mode(n=32, *, dtype=torch.complex128):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    t = torch.linspace(0.0, 1.0, n, dtype=real_dtype)
    return torch.exp(torch.complex(torch.ones_like(t) * 0.1, 2.0 * t))


@pytest.mark.parametrize("inclination,phic", [(0.0, 0.0), (0.7, 0.0), (0.0, 0.4), (0.7, 0.4)])
def test_generic_and_direct_projection_cancel(inclination, phic):
    mode = _mode()
    generic = adapter.project_teob22_generic(
        mode,
        mass1=30.0,
        mass2=20.0,
        distance=100.0,
        inclination=inclination,
        phic=phic,
    )
    direct = adapter.project_teob22_direct(
        mode,
        mass1=30.0,
        mass2=20.0,
        distance=100.0,
        inclination=inclination,
        phic=phic,
    )
    assert torch.max(torch.abs(generic[0] + direct[0])).item() <= 1.3e-16
    assert torch.max(torch.abs(generic[1] + direct[1])).item() <= 1.3e-16


def test_projection_supports_extrinsic_batches_and_gradients():
    mode = _mode(12)
    inclination = torch.tensor([0.0, 0.7], dtype=torch.float64, requires_grad=True)
    phic = torch.tensor([0.0, 0.4], dtype=torch.float64, requires_grad=True)
    plus, cross = adapter.project_teob22_generic(
        mode,
        mass1=torch.tensor([30.0, 30.0], dtype=torch.float64, requires_grad=True),
        mass2=torch.tensor([20.0, 20.0], dtype=torch.float64),
        distance=100.0,
        inclination=inclination,
        phic=phic,
    )
    assert plus.shape == (2, mode.numel())
    assert cross.shape == plus.shape
    (plus.square().sum() + cross.square().sum()).backward()
    assert torch.isfinite(inclination.grad).all()
    assert torch.isfinite(phic.grad).all()


def test_policies_mode_and_intrinsic_domain_are_explicit():
    with pytest.raises(TypeError, match="required explicitly"):
        adapter.conditioned_teob_fd(30.0, 20.0, f_low=20.0, f_max=256.0, delta_f=4.0)
    with pytest.raises(NotImplementedError, match=r"mode \(2,2\)"):
        adapter.conditioned_teob_fd(
            30.0,
            20.0,
            f_low=20.0,
            f_max=256.0,
            delta_f=4.0,
            radial_acceleration_policy="conservative",
            horizon_policy="current",
            mode=(3, 3),
        )
    with pytest.raises(NotImplementedError, match="zero aligned spins"):
        adapter.conditioned_teob_fd(
            30.0,
            20.0,
            f_low=20.0,
            f_max=256.0,
            delta_f=4.0,
            chi1z=0.1,
            radial_acceleration_policy="conservative",
            horizon_policy="current",
        )
    with pytest.raises(ValueError, match="unsupported horizon_policy"):
        adapter.conditioned_teob_fd(
            30.0,
            20.0,
            f_low=20.0,
            f_max=256.0,
            delta_f=4.0,
            radial_acceleration_policy="conservative",
            horizon_policy="fitted",
        )
    with pytest.raises(ValueError, match="unsupported remnant_policy"):
        adapter.conditioned_teob_fd(
            30.0,
            20.0,
            f_low=20.0,
            f_max=256.0,
            delta_f=4.0,
            radial_acceleration_policy="conservative",
            horizon_policy="current",
            remnant_policy="unknown",
        )


def test_leading_order_horizon_force_is_finite_and_differentiable():
    dtype = torch.float64
    mass1 = torch.tensor(30.0, dtype=dtype, requires_grad=True)
    mass2 = torch.tensor(20.0, dtype=dtype)
    ham = adapter._make_hamiltonian(mass1, mass2)
    reaction = adapter.LeadingOrderHorizonRadiationReaction()
    radius = torch.tensor([18.0, 16.0], dtype=dtype, requires_grad=True)
    p_r = torch.tensor([0.0, -1.0e-3], dtype=dtype)
    p_phi = torch.tensor([4.5, 4.3], dtype=dtype)
    omega = torch.tensor([0.014, 0.016], dtype=dtype)
    radial, torque = reaction.radiation_reaction_forces(
        omega, p_r, p_phi, r=radius, hamiltonian=ham,
    )
    assert torch.equal(radial, torch.zeros_like(radial))
    assert torch.isfinite(torque).all()
    torque.sum().backward()
    assert mass1.grad is not None and torch.isfinite(mass1.grad)
    assert radius.grad is not None and torch.isfinite(radius.grad).all()


@pytest.mark.parametrize("matching_method", ["grid", "continuous"])
def test_orchestration_passes_native_radius_and_named_policies(monkeypatch, matching_method):
    dtype = torch.float64
    captured = {}
    n = 16
    time = torch.linspace(0.0, 1.5, n, dtype=dtype)
    state = torch.stack(
        (
            torch.linspace(18.0, 14.0, n, dtype=dtype),
            torch.linspace(0.0, 0.2, n, dtype=dtype),
            torch.linspace(0.0, -1.0e-3, n, dtype=dtype),
            torch.linspace(4.5, 4.2, n, dtype=dtype),
        ),
        dim=-1,
    )
    frequency = torch.linspace(0.013, 0.019, n, dtype=dtype)
    fake_pa = SimpleNamespace(
        complete=True,
        status="complete",
        skip_reason=None,
        endpoint_state=state[-1],
        endpoint_time=time[-1],
    )
    fake_continuation = SimpleNamespace(
        complete=True,
        status="complete",
        skip_reason=None,
        time=time,
        state=state,
        frequency=frequency,
        fixed_step=torch.tensor(0.1, dtype=dtype),
        peak_event_time=time[3],
    )
    fake_series = SimpleNamespace(
        time=time,
        corrected_amplitude=torch.ones(n, dtype=dtype),
        corrected_phase=torch.linspace(0.0, 1.0, n, dtype=dtype),
        auxiliary_orbital_frequency=frequency[:8],
    )
    fake_nqc = SimpleNamespace(
        original=fake_series,
        auxiliary=SimpleNamespace(
            time=time[:8],
            corrected_amplitude=torch.ones(8, dtype=dtype),
            corrected_phase=torch.linspace(0.0, 0.5, 8, dtype=dtype),
            orbital_frequency=frequency[:8],
        ),
        matching_index=5,
        t_peak=time[5],
    )
    fake_ringdown = SimpleNamespace(
        mode=_mode(n),
        time=time,
        epoch=time[0],
        events=SimpleNamespace(peak_time=time[5]),
    )
    remnant_inputs = []

    def fake_pa_builder(ham, rr, **kwargs):
        captured["pa"] = kwargs
        return fake_pa

    def fake_cont_builder(ham, pa, rr, **kwargs):
        captured["continuation"] = kwargs
        return fake_continuation

    monkeypatch.setattr(adapter, "alternating8pa", fake_pa_builder)
    monkeypatch.setattr(adapter, "continue_teob_adaptive", fake_cont_builder)
    monkeypatch.setattr(
        adapter,
        "conservative_trajectory_quantities",
        lambda ham, state: SimpleNamespace(radial_acceleration=torch.zeros(n, dtype=dtype)),
    )
    monkeypatch.setattr(adapter, "teob_waveform_state_mode", lambda *args, **kwargs: _mode(n))
    def fake_nqc_builder(*args, **kwargs):
        captured["nqc"] = kwargs
        return fake_nqc

    monkeypatch.setattr(adapter, "assemble_teob_merger_nqc", fake_nqc_builder)
    def fake_ringdown_builder(*args, **kwargs):
        # The ringdown contract passes final mass and spin after nu.  Capture
        # both to verify that the adapter selects the requested model version.
        remnant_inputs.append((args[8], args[9]))
        captured["ringdown"] = kwargs
        return fake_ringdown

    monkeypatch.setattr(adapter, "assemble_teob_ringdown_22", fake_ringdown_builder)

    def fake_condition(hp, hc, **kwargs):
        sampling = fft_length_and_sampling(kwargs["f_max"], kwargs["delta_f"])
        captured["conditioning"] = kwargs
        return ConditionedFD(
            frequencies=sampling.frequencies(),
            h_plus=hp,
            h_cross=hc,
            sampling=sampling,
            conditioning=SimpleNamespace(),
            tc=kwargs["tc"],
        )

    monkeypatch.setattr(adapter, "condition_td_to_fd", fake_condition)
    result = adapter.conditioned_teob_fd(
        30.0,
        20.0,
        f_low=20.0,
        f_max=256.0,
        delta_f=4.0,
        radial_acceleration_policy="zero",
        horizon_policy="leading-order",
        pa_kwargs={"r_min": 14.0, "dr": 0.2, "pa_order": 8},
        continuation_kwargs={"continuation_duration": 2.0},
        matching_method=matching_method,
    )
    expected = torch.maximum(
        torch.tensor(14.0, dtype=dtype),
        result.omega_start.pow(-2.0 / 3.0),
    )
    assert torch.equal(result.initial_radius, expected)
    assert captured["pa"]["omega_start"] is result.omega_start
    assert captured["pa"]["initializer_policy"] == "analytic"
    assert result.radial_acceleration_policy == "zero"
    assert result.horizon_policy == "leading-order"
    assert result.remnant_policy == "uib2016"
    assert result.nqc_target_policy == "published-2017"
    assert result.metadata["matching_method"] == matching_method
    assert captured["nqc"]["matching_method"] == matching_method
    assert captured["ringdown"]["matching_method"] == matching_method
    if matching_method == "continuous":
        assert captured["ringdown"]["peak_time"] is fake_nqc.t_peak
    else:
        assert captured["ringdown"]["peak_time"] is None
    expected_uib = adapter.uib2016_nonspinning_remnant(30.0, 20.0)
    torch.testing.assert_close(remnant_inputs[0][0], expected_uib[0])
    torch.testing.assert_close(remnant_inputs[0][1], expected_uib[1])
    assert result.fd.frequencies.shape == result.frequencies.shape
    assert captured["conditioning"]["dt"].shape == torch.Size([])
    assert torch.equal(
        result.physical_epoch_seconds,
        (result.td_time[0] - fake_continuation.peak_event_time)
        * result.total_mass_seconds,
    )
    assert not torch.equal(
        result.physical_epoch_seconds,
        (result.td_time[0] - fake_ringdown.events.peak_time)
        * result.total_mass_seconds,
    )

    frozen_inputs = {}

    def fake_frozen(ham, radius, pr_star, p_phi):
        frozen_inputs.update(radius=radius, pr_star=pr_star, p_phi=p_phi)
        return torch.full_like(radius, 0.25)

    monkeypatch.setattr(adapter, "frozen_mobility_radial_acceleration", fake_frozen)
    frozen_result = adapter.conditioned_teob_fd(
        30.0,
        20.0,
        f_low=20.0,
        f_max=256.0,
        delta_f=4.0,
        radial_acceleration_policy="frozen-mobility",
        horizon_policy="leading-order",
        remnant_policy="teob-healy-jf",
        pa_initializer_policy="quadratic-value",
        pa_kwargs={"r_min": 14.0, "dr": 0.2, "pa_order": 8},
        continuation_kwargs={"continuation_duration": 2.0},
    )
    assert frozen_result.radial_acceleration_policy == "frozen-mobility"
    assert frozen_result.remnant_policy == "teob-healy-jf"
    assert frozen_result.pa_initializer_policy == "quadratic-value"
    assert captured["pa"]["initializer_policy"] == "quadratic-value"
    expected_hybrid = adapter.teob_hybrid_nonspinning_remnant(30.0, 20.0)
    torch.testing.assert_close(remnant_inputs[1][0], expected_hybrid[0])
    torch.testing.assert_close(remnant_inputs[1][1], expected_hybrid[1])
    assert torch.equal(frozen_inputs["radius"], state[:, 0])
    assert torch.equal(frozen_inputs["pr_star"], state[:, 2])
    assert torch.equal(frozen_inputs["p_phi"], state[:, 3])

    direct_result = adapter.conditioned_teob_fd(
        30.0,
        20.0,
        f_low=20.0,
        f_max=256.0,
        delta_f=4.0,
        radial_acceleration_policy="frozen-mobility",
        horizon_policy="leading-order",
        pa_initializer_policy="direct-quadratic-value",
        pa_kwargs={"r_min": 14.0, "dr": 0.2, "pa_order": 8},
        continuation_kwargs={"continuation_duration": 2.0},
    )
    assert direct_result.pa_initializer_policy == "direct-quadratic-value"
    assert captured["pa"]["initializer_policy"] == "direct-quadratic-value"


def test_native_support_start_controls_inverse_kepler_radius():
    support = plan_td_support(30.0, 20.0, 20.0, dtype=torch.float64)
    omega = torch.pi * support.total_mass_seconds * support.start_frequency
    assert torch.allclose(omega.pow(-2.0 / 3.0), torch.tensor(18.01749118199749, dtype=torch.float64))


def test_stage_aware_frozen_mobility_scales_only_pa_prefix():
    dtype = torch.float64
    ham = adapter._make_hamiltonian(
        torch.tensor(30.0, dtype=dtype), torch.tensor(20.0, dtype=dtype),
    )
    state = torch.tensor([
        [18.0, 0.0, -1.0e-3, 4.5],
        [17.0, 0.1, -2.0e-3, 4.4],
        [16.0, 0.2, -3.0e-3, 4.3],
        [15.0, 0.3, -4.0e-3, 4.2],
    ], dtype=dtype)
    continuation = SimpleNamespace(
        state=state,
        pa_time=torch.tensor([0.0, 0.1], dtype=dtype),
        pa_state=state[:2],
    )
    base = adapter.frozen_mobility_radial_acceleration(
        ham, state[:, 0], state[:, 2], state[:, 3],
    )
    result = adapter._stage_aware_frozen_mobility_radial_acceleration(ham, continuation)
    energy = ham.real_hamiltonian(state[:2, 0], state[:2, 2], state[:2, 3])
    torch.testing.assert_close(result[:2], base[:2] * energy.square())
    torch.testing.assert_close(result[2:], base[2:])


def test_stage_aware_frozen_mobility_without_pa_is_unchanged():
    dtype = torch.float64
    ham = adapter._make_hamiltonian(torch.tensor(30.0, dtype=dtype), torch.tensor(20.0, dtype=dtype))
    state = torch.tensor([[18.0, 0.0, -1.0e-3, 4.5], [17.0, 0.1, -2.0e-3, 4.4]], dtype=dtype)
    continuation = SimpleNamespace(state=state, pa_time=None, pa_state=None)
    expected = adapter.frozen_mobility_radial_acceleration(ham, state[:, 0], state[:, 2], state[:, 3])
    torch.testing.assert_close(adapter._stage_aware_frozen_mobility_radial_acceleration(ham, continuation), expected)


def test_stage_aware_frozen_mobility_rejects_inconsistent_pa_shape():
    dtype = torch.float64
    ham = adapter._make_hamiltonian(torch.tensor(30.0, dtype=dtype), torch.tensor(20.0, dtype=dtype))
    state = torch.zeros((3, 4), dtype=dtype)
    state[:, 0] = 16.0
    state[:, 3] = 4.0
    continuation = SimpleNamespace(
        state=state, pa_time=torch.tensor([0.0, 0.1], dtype=dtype), pa_state=state[:1],
    )
    with pytest.raises(adapter.TEOBConditionedWaveformError, match="PA prefix"):
        adapter._stage_aware_frozen_mobility_radial_acceleration(ham, continuation)
