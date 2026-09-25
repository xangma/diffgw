"""Cross-backend checks for the development-only adaptive RKF45 core."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom.eob.jax.adaptive_driver_clean import (
    AlignedSpinV4Event as JaxEvent,
    fixed_fehlberg45_step,
    integrate_adaptive_to_time,
    replay_accepted_steps,
    resample_accepted_trajectory,
    state_scaled_controller_adjustment,
)
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from torchwave.approximants.eob.adaptive_driver_clean import (
    AlignedSpinV4Event as TorchEvent,
    fixed_fehlberg45_step as torch_fixed_step,
    integrate_adaptive as torch_integrate_adaptive,
    resample_natural_cubic as torch_resample_natural_cubic,
    state_scaled_controller_adjustment as torch_controller,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxEOBIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as JaxHam
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchEOBIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian as TorchHam

jax.config.update("jax_enable_x64", True)


def _jax_field(time, state):
    return jnp.stack((state[..., 0], 0.02 * jnp.sin(time) + 0.03 * jnp.ones_like(time)), axis=-1)


def _torch_field(time, state):
    return torch.stack((state[..., 0], 0.02 * torch.sin(time) + 0.03 * torch.ones_like(time)), dim=-1)


def test_fixed_trial_torch_parity_large_phase_and_jit():
    offsets = jnp.asarray([0.0, 132.0, 1e6, 1e10])
    state = jnp.stack((jnp.ones_like(offsets), offsets), axis=-1)
    time = jnp.full_like(offsets, 0.3)
    step = jnp.full_like(offsets, 0.1)
    actual = jax.jit(lambda t, y, h: fixed_fehlberg45_step(
        _jax_field, t, y, h, atol=1e-13, rtol=1e-11
    ))(time, state, step)
    expected = [torch_fixed_step(
        _torch_field, torch.tensor(float(t), dtype=torch.float64),
        torch.tensor(np.asarray(y), dtype=torch.float64),
        torch.tensor(float(h), dtype=torch.float64), atol=1e-13, rtol=1e-11,
    ) for t, y, h in zip(np.asarray(time), np.asarray(state), np.asarray(step))]
    for name in ("fourth", "fifth", "error", "error_ratio"):
        reference = np.stack([getattr(row, name).detach().numpy() for row in expected])
        tolerance = 1e-9 if name == "error_ratio" else 2e-12
        np.testing.assert_allclose(np.asarray(getattr(actual, name)), reference,
                                   rtol=tolerance, atol=2e-15)
    np.testing.assert_allclose(np.asarray(actual.error[:, 1]),
                               np.full(4, float(actual.error[0, 1])), rtol=0, atol=2e-18)
    assert abs(float(actual.error[0, 1])) > 1e-13


def test_state_scaled_controller_matches_torch_thresholds():
    ratios = jnp.asarray([0.0, 0.1, 0.5, 0.8, 1.1, 2.0, 1e5])
    action, factor = jax.jit(state_scaled_controller_adjustment)(ratios)
    other_action, other_factor = torch_controller(torch.tensor(np.asarray(ratios)))
    np.testing.assert_array_equal(np.asarray(action), other_action.numpy())
    np.testing.assert_allclose(np.asarray(factor), other_factor.numpy(), rtol=2e-15)


def test_accepted_history_matches_torch_and_retains_gradients():
    def field_jax(t, y):
        return jnp.stack((y[0], -0.2 * y[1] + jnp.sin(t)))

    def field_torch(t, y):
        return torch.stack((y[..., 0], -0.2 * y[..., 1] + torch.sin(t)), dim=-1)

    initial = jnp.asarray([1.0, 0.3])
    run = integrate_adaptive_to_time(field_jax, initial, 1.0, initial_step=0.1,
                                     atol=1e-10, rtol=1e-9, max_step=0.2)
    reference = torch_integrate_adaptive(
        field_torch, torch.tensor([1.0, 0.3], dtype=torch.float64), 0.1,
        t_end=1.0, initial_step=0.1, max_step=0.2, atol=1e-10, rtol=1e-9,
        controller="gsl_state_scaled",
    )
    np.testing.assert_allclose(np.asarray(run.time), reference.time.detach().numpy(), rtol=0, atol=2e-11)
    np.testing.assert_allclose(np.asarray(run.state), reference.state.detach().numpy(), rtol=2e-10, atol=2e-11)
    assert run.accepted_steps == int(reference.statistics.accepted_steps)
    assert run.rejected_steps == int(reference.statistics.rejected_steps)
    assert run.rejected_steps > 0
    assert run.evaluations == int(reference.statistics.evaluations)
    assert run.termination == "duration"
    grid, sampled = resample_accepted_trajectory(run, 0.07)
    expected_grid, expected_sampled = torch_resample_natural_cubic(
        reference.time, reference.state, 0.07)
    np.testing.assert_allclose(np.asarray(grid), expected_grid.detach().numpy(), rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(sampled), expected_sampled.detach().numpy(), rtol=2e-10, atol=2e-11)
    replay = jax.jit(lambda state: replay_accepted_steps(field_jax, state, run.time))(initial)
    np.testing.assert_allclose(np.asarray(replay), np.asarray(run.state), rtol=2e-12, atol=2e-12)
    derivative = jax.grad(lambda first: replay_accepted_steps(
        field_jax, jnp.stack((first, jnp.asarray(0.3))), run.time,
    )[-1, 0])(jnp.asarray(1.0))
    np.testing.assert_allclose(float(derivative), np.e, rtol=2e-7)


def test_one_physical_eob_fehlberg_trial_matches_torch():
    ham_jax = JaxHam(0.24, 0.2, -0.1, delta=0.2)
    ham_torch = TorchHam(torch.tensor(0.24, dtype=torch.float64),
                         torch.tensor(0.2, dtype=torch.float64),
                         torch.tensor(-0.1, dtype=torch.float64),
                         delta=torch.tensor(0.2, dtype=torch.float64))
    field_jax = JaxEOBIntegrator(ham_jax, JaxFlux(spin_truncation="tabulated")).vector_field
    field_torch = TorchEOBIntegrator(ham_torch, TorchFlux(spin_truncation="tabulated")).vector_field
    state = jnp.asarray([12.0, 0.0, -0.02, 3.8])
    actual = fixed_fehlberg45_step(lambda _t, y: field_jax(y), 0.0, state, 0.1)
    expected = torch_fixed_step(lambda _t, y: field_torch(y), 0.0,
                                torch.tensor(np.asarray(state), dtype=torch.float64), 0.1)
    for key in ("fifth", "error"):
        np.testing.assert_allclose(np.asarray(getattr(actual, key)),
                                   getattr(expected, key).detach().numpy(), rtol=2e-9, atol=2e-12)
    run = integrate_adaptive_to_time(lambda _t, y: field_jax(y), state, 0.5,
                                     initial_step=0.1, max_step=0.25)
    reference = torch_integrate_adaptive(
        lambda _t, y: field_torch(y),
        torch.tensor(np.asarray(state), dtype=torch.float64), 0.1,
        t_end=0.5, initial_step=0.1, max_step=0.25,
        controller="gsl_state_scaled",
    )
    assert run.accepted_steps == int(reference.statistics.accepted_steps)
    assert run.rejected_steps == int(reference.statistics.rejected_steps)
    np.testing.assert_allclose(np.asarray(run.time), reference.time.detach().numpy(), rtol=0, atol=2e-10)
    np.testing.assert_allclose(np.asarray(run.state), reference.state.detach().numpy(), rtol=2e-9, atol=2e-11)
    grid, sampled = resample_accepted_trajectory(run, 0.05)
    expected_grid, expected_sampled = torch_resample_natural_cubic(
        reference.time, reference.state, 0.05)
    np.testing.assert_allclose(np.asarray(grid), expected_grid.detach().numpy(), rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(sampled), expected_sampled.detach().numpy(), rtol=2e-9, atol=2e-11)


def test_accepted_knot_fine_event_matches_torch():
    def field_jax(t, state):
        return jnp.stack((-jnp.ones_like(t), jnp.sin(t),
                          -0.1 * jnp.ones_like(t), jnp.zeros_like(t)))

    def field_torch(t, state):
        return torch.stack((-torch.ones_like(t), torch.sin(t),
                            -0.1 * torch.ones_like(t), torch.zeros_like(t)), dim=-1)

    initial = [6.5, 0.0, -0.02, 3.8]
    run = integrate_adaptive_to_time(field_jax, jnp.asarray(initial), 3.0,
                                     initial_step=0.1, max_step=0.1,
                                     atol=1e-8, rtol=1e-7,
                                     event=JaxEvent(stage="fine", frequency_decreases=3))
    reference = torch_integrate_adaptive(
        field_torch, torch.tensor(initial, dtype=torch.float64), 0.1,
        t_end=3.0, initial_step=0.1, max_step=0.1,
        atol=1e-8, rtol=1e-7, controller="gsl_state_scaled",
        event=TorchEvent(stage="fine", frequency_decreases=3),
    )
    assert run.termination == reference.statistics.termination[0] == "event"
    np.testing.assert_allclose(np.asarray(run.time), reference.time.detach().numpy(), rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.asarray(run.state), reference.state.detach().numpy(), rtol=1e-12, atol=1e-12)


def test_two_stage_restart_preserves_phase_and_matches_torch_grid():
    def field_jax(state):
        return jnp.stack((-0.2 * jnp.ones_like(state[0]),
                          0.01 * state[0], -0.01 * jnp.ones_like(state[0]),
                          jnp.zeros_like(state[0])))

    def field_torch(state):
        radius = state[..., 0]
        return torch.stack((-0.2 * torch.ones_like(radius),
                            0.01 * radius, -0.01 * torch.ones_like(radius),
                            torch.zeros_like(radius)), dim=-1)

    initial = [10.0, 10.0, -0.02, 3.8]
    actual = integrate_two_stage_trajectory(
        field_jax, jnp.asarray(initial), output_step=0.5,
        max_duration=100.0, coarse_max_step=0.5, fine_max_step=0.5,
    )
    coarse = torch_integrate_adaptive(
        field_torch, torch.tensor(initial, dtype=torch.float64), 0.5,
        t_end=100.0, event=TorchEvent("coarse"), initial_step=0.5,
        max_step=0.5, atol=1e-10, rtol=1e-9,
        controller="gsl_state_scaled",
    )
    n_coarse = len(coarse.output_time)
    restart_index = n_coarse - (n_coarse + 1) // 2
    fine_step = 0.5 / 32
    fine = torch_integrate_adaptive(
        field_torch, coarse.output_state[restart_index], fine_step,
        t_end=100.0, event=TorchEvent("fine"), initial_step=fine_step,
        max_step=0.5, atol=1e-10, rtol=1e-9,
        controller="gsl_state_scaled",
    )
    assert actual.coarse.termination == actual.fine.termination == "event"
    assert actual.restart_index == restart_index
    assert actual.refinement_factor == 32
    assert actual.coarse.accepted_steps == int(coarse.statistics.accepted_steps)
    assert actual.fine.accepted_steps == int(fine.statistics.accepted_steps)
    np.testing.assert_allclose(np.asarray(actual.fine.state[0]),
                               coarse.output_state[restart_index].detach().numpy(),
                               rtol=0, atol=2e-12)
    expected_time = torch.cat((coarse.output_time[:restart_index],
                               coarse.output_time[restart_index] + fine.output_time[::32]))
    expected_state = torch.cat((coarse.output_state[:restart_index],
                                fine.output_state[::32]))
    np.testing.assert_allclose(np.asarray(actual.time), expected_time.detach().numpy(),
                               rtol=0, atol=2e-11)
    np.testing.assert_allclose(np.asarray(actual.state), expected_state.detach().numpy(),
                               rtol=2e-10, atol=2e-11)
