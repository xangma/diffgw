"""Development-only prototype; scientific qualification is incomplete.

Independent bounded Fehlberg 4/5 drivers for the clean EOB route.

The adaptive state history in this module is deliberately separate from the
uniform waveform grid.  Accepted states are advanced with the embedded
Fehlberg pair from NASA TR R-315; a natural cubic interpolant is then used to
sample the requested output spacing.  The event tests operate on accepted
states, never on the interpolated samples.

The generic :func:`integrate_adaptive` routine accepts a state-only callable
(``field(state)``) or a time-dependent callable (``field(time, state)``).  It
supports leading batch axes, preserves the Torch graph through all accepted
arithmetic, and makes the discrete controller and event decisions explicit.
The EOB-specific two-stage helper is opt-in and is intentionally kept out of
the default fixed-step published route.
"""

from __future__ import annotations
from dataclasses import replace
import math
from typing import Optional
import torch
from torch import Tensor
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.numerical_attachment import (
    attach_seobnrv4_22_numerical_compatibility,
)
from torchwave.approximants.eob.nqc import nqc_22_basis

from torchwave.approximants.eob.adaptive_driver_clean import (
    GSL_STATE_SCALED_CONTROLLER,
    AdaptiveIntegrationError,
    integrate_adaptive,
    AlignedSpinV4Event,
    SplicedNumericalAttachment,
    TwoStageEOBTrajectory,
    _stack_adaptive_trajectories,
    _preserve_output_phase,
    _apply_calibrated_nqc,
    _rkf45_trial,
)


def _diagnostic_hermite_output(run):
    """Resample one scalar accepted trajectory with its accepted derivatives."""
    if run.time.ndim != 1 or run.state.ndim != 2 or run.time.numel() < 2:
        raise ValueError("diagnostic Hermite resampling requires scalar accepted states")
    query = run.output_time
    left = torch.searchsorted(run.time.detach(), query.detach(), right=True) - 1
    left = left.clamp(0, run.time.numel() - 2)
    t0, t1 = run.time[left], run.time[left + 1]
    h = t1 - t0
    u = (query - t0) / h
    u2, u3 = u * u, u * u * u
    y0, y1 = run.state[left], run.state[left + 1]
    f0, f1 = run.derivative[left], run.derivative[left + 1]
    sample = ((2*u3 - 3*u2 + 1).unsqueeze(-1) * y0
              + ((u3 - 2*u2 + u) * h).unsqueeze(-1) * f0
              + (-2*u3 + 3*u2).unsqueeze(-1) * y1
              + ((u3 - u2) * h).unsqueeze(-1) * f1)
    return replace(run, output_state=sample)


def _diagnostic_rkf45_output(run, vector_field):
    """Sample a scalar accepted trajectory by fixed Fehlberg substeps.

    Each output query starts at its preceding accepted state.  This changes
    only output sampling: accepted states, derivatives, event times and solver
    statistics remain those of the original adaptive run.  The diagnostic is
    deliberately separate from the published reconstruction path.
    """
    if run.time.ndim != 1 or run.state.ndim != 2 or run.time.numel() < 2:
        raise ValueError("diagnostic Fehlberg resampling requires scalar accepted states")
    query = run.output_time
    left = torch.searchsorted(run.time.detach(), query.detach(), right=True) - 1
    left = left.clamp(0, run.time.numel() - 2)
    step = query - run.time[left]
    if not bool(((step >= 0) & (step <= run.time[left + 1] - run.time[left])).all()):
        raise ValueError("output query lies outside its accepted step")
    _, sample, _, _ = _rkf45_trial(
        vector_field, run.time[left], run.state[left], step,
        run.derivative[left], 1,
    )
    sample = torch.where((query == run.time[-1]).unsqueeze(-1),
                         run.state[-1], sample)
    return replace(run, output_state=sample)


def _diagnostic_local_nqc_phase_matrix(time, state, orbital_frequency, match_time,
                                       window=0.5):
    """Degree-six local NQC derivative estimate for numerical diagnosis only.

    The default 0.5 M window was selected after comparison with dense DOP853
    differentiation on one nonspinning row. It is not a published v4 rule;
    other windows are numerical diagnostics only.
    """
    selected = (time - match_time).detach().abs() <= window
    chart = (time[selected] - match_time) / window
    if chart.numel() < 9 or not bool((chart.min() < 0) & (chart.max() > 0)):
        raise ValueError("local NQC phase derivative requires two-sided samples")
    _, basis = nqc_22_basis(state[selected, 0], state[selected, 2],
                            orbital_frequency[selected])
    design = torch.stack([chart.pow(k) for k in range(7)], dim=-1)
    if not bool(torch.isfinite(design).all()) or not bool(torch.isfinite(basis).all()):
        raise ValueError("nonfinite local NQC phase derivative input")
    polynomial = torch.linalg.lstsq(design, basis, driver="gels").solution
    return torch.stack((polynomial[1] / window, 2 * polynomial[2] / window**2))


def _diagnostic_local_matching_second(time, nqc, match_time):
    """Predeclared local second derivatives for five NQC matching inputs."""
    window = 0.5
    selected = (time - match_time).detach().abs() <= window
    chart = (time[selected] - match_time) / window
    if chart.numel() < 9 or not bool((chart.min() < 0) & (chart.max() > 0)):
        raise ValueError("local NQC second derivative requires two-sided samples")
    values = torch.cat((nqc.sampled_amplitude.unsqueeze(-1),
                        nqc.sampled_products,
                        nqc.sampled_waveform_phase.unsqueeze(-1)), dim=-1)[selected]
    design = torch.stack([chart.pow(k) for k in range(7)], dim=-1)
    if not bool(torch.isfinite(design).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite local NQC second derivative input")
    polynomial = torch.linalg.lstsq(
        design, values - values[chart.numel() // 2], driver="gels").solution
    return 2 * polynomial[2] / window**2


def reconstruct_seobnrv4_22_adaptive(
    omega_start: Tensor,
    mass1,
    mass2,
    chi1=0.0,
    chi2=0.0,
    *,
    spin_truncation: str = "generic-leading-ell8",
    diagnostic_mode_21_higher_spin: str = "taylor_f",
    diagnostic_mode_31_higher_spin: str = "taylor_f",
    diagnostic_mode_32_spin_square: str = "pan_f26",
    diagnostic_mode_33_higher_spin: str = "taylor_f",
    diagnostic_mode_43_higher_spin: str = "taylor_f",
    diagnostic_hamiltonian_log_derivative: str = "analytic",
    diagnostic_hamiltonian_calibration: str = "published",
    spin_v6: str = "bohe_total_energy",
    include_pan_v8: bool = False,
    include_positive_chi_540: bool = False,
    final_spin_fit_order=(3, 4),
    dt_geom: float = 0.5,
    diagnostic_coarse_initial_step_M: Optional[float] = None,
    diagnostic_warm_start_Momega: Optional[float] = None,
    diagnostic_restart_time_M: Optional[float] = None,
    diagnostic_fine_max_step_M: Optional[float] = None,
    diagnostic_hermite_resampling: bool = False,
    diagnostic_rkf45_resampling: bool = False,
    diagnostic_local_nqc_derivatives: bool = False,
    diagnostic_nqc_phase_window_M: float = 0.5,
    diagnostic_matching_second_derivatives: bool = False,
    num_steps: int = 20000,
    ringdown_duration: float = 150.0,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-9,
    max_rejected_steps: int = 10000,
    max_evaluations: int = 200000,
    max_accepted_steps: Optional[int] = None,
    max_duration: Optional[float] = None,
):
    """Run the opt-in two-stage numerical-compatibility EOB reconstruction.

    The coarse stage starts at ``dt_geom`` and is naturally resampled on that
    grid.  A fine restart is selected from a real coarse sample using the
    150M/half-trajectory rule; its local clock starts at zero while the
    restart state's orbital phase is retained.  The returned mode is the
    coarse prefix followed by every power-of-two-th fine sample.  This helper
    is intentionally independent of the fixed-step default reconstruction.
    The diagnostic overrides isolate solver-start and restart-sample effects
    in grid-convergence studies; neither changes the published default.
    Hermite resampling is a separate opt-in numerical experiment.
    """
    if not isinstance(omega_start, Tensor) or omega_start.dtype not in (torch.float32, torch.float64):
        raise TypeError("omega_start must be a float32/float64 tensor")
    if not math.isfinite(float(dt_geom)) or dt_geom <= 0:
        raise ValueError("dt_geom must be finite and positive")
    if (diagnostic_coarse_initial_step_M is not None
            and (not math.isfinite(float(diagnostic_coarse_initial_step_M))
                 or diagnostic_coarse_initial_step_M <= 0)):
        raise ValueError("diagnostic_coarse_initial_step_M must be finite and positive")
    if (diagnostic_warm_start_Momega is not None
            and (not math.isfinite(float(diagnostic_warm_start_Momega))
                 or diagnostic_warm_start_Momega <= 0)):
        raise ValueError("diagnostic_warm_start_Momega must be finite and positive")
    if (diagnostic_restart_time_M is not None
            and (not math.isfinite(float(diagnostic_restart_time_M))
                 or diagnostic_restart_time_M <= 0)):
        raise ValueError("diagnostic_restart_time_M must be finite and positive")
    if (diagnostic_fine_max_step_M is not None
            and (not math.isfinite(float(diagnostic_fine_max_step_M))
                 or diagnostic_fine_max_step_M <= 0)):
        raise ValueError("diagnostic_fine_max_step_M must be finite and positive")
    if (diagnostic_restart_time_M is not None
            and not math.isclose(round(diagnostic_restart_time_M / dt_geom) * dt_geom,
                                 diagnostic_restart_time_M,
                                 rel_tol=0.0, abs_tol=1.0e-10)):
        raise ValueError("diagnostic_restart_time_M must lie on the output grid")
    if not isinstance(diagnostic_hermite_resampling, bool):
        raise TypeError("diagnostic_hermite_resampling must be bool")
    if not isinstance(diagnostic_rkf45_resampling, bool):
        raise TypeError("diagnostic_rkf45_resampling must be bool")
    if diagnostic_hermite_resampling and diagnostic_rkf45_resampling:
        raise ValueError("choose one diagnostic output resampler")
    if not isinstance(diagnostic_local_nqc_derivatives, bool):
        raise TypeError("diagnostic_local_nqc_derivatives must be bool")
    if (not math.isfinite(float(diagnostic_nqc_phase_window_M))
            or diagnostic_nqc_phase_window_M <= 0):
        raise ValueError("diagnostic_nqc_phase_window_M must be finite and positive")
    if diagnostic_nqc_phase_window_M != 0.5 and not diagnostic_local_nqc_derivatives:
        raise ValueError("diagnostic_nqc_phase_window_M requires local NQC derivatives")
    if not isinstance(diagnostic_matching_second_derivatives, bool):
        raise TypeError("diagnostic_matching_second_derivatives must be bool")
    if not isinstance(num_steps, int) or num_steps < 4:
        raise ValueError("num_steps must be an integer of at least four")
    if not math.isfinite(float(ringdown_duration)) or ringdown_duration <= 0:
        raise ValueError("ringdown_duration must be finite and positive")
    if spin_truncation not in {"generic-leading-ell8", "tabulated"}:
        raise ValueError("spin_truncation must be generic-leading-ell8 or tabulated")
    if diagnostic_mode_21_higher_spin not in {"taylor_f", "rho_hybrid"}:
        raise ValueError("diagnostic_mode_21_higher_spin must be taylor_f or rho_hybrid")
    if diagnostic_mode_31_higher_spin not in {"taylor_f", "lal_v4_rho"}:
        raise ValueError("diagnostic_mode_31_higher_spin must be taylor_f or lal_v4_rho")
    if diagnostic_mode_32_spin_square not in {"pan_f26", "lal_v4_zero"}:
        raise ValueError("diagnostic_mode_32_spin_square must be pan_f26 or lal_v4_zero")
    if diagnostic_mode_33_higher_spin not in {"taylor_f", "lal_v4_rho"}:
        raise ValueError("diagnostic_mode_33_higher_spin must be taylor_f or lal_v4_rho")
    if diagnostic_mode_43_higher_spin not in {"taylor_f", "lal_v4_rho"}:
        raise ValueError("diagnostic_mode_43_higher_spin must be taylor_f or lal_v4_rho")
    if diagnostic_hamiltonian_log_derivative not in {"analytic", "lal_v4"}:
        raise ValueError("diagnostic_hamiltonian_log_derivative must be analytic or lal_v4")
    if diagnostic_hamiltonian_calibration not in {"published", "lal_v4_rounded"}:
        raise ValueError("diagnostic_hamiltonian_calibration must be published or lal_v4_rounded")
    if tuple(final_spin_fit_order) != (3, 4):
        raise ValueError(
            "adaptive numerical compatibility requires final_spin_fit_order=(3, 4)"
        )
    if spin_v6 != "bohe_total_energy" or include_pan_v8 is not False or include_positive_chi_540 is not False:
        raise ValueError("adaptive numerical compatibility requires the Bohé total-energy phase choices")
    from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
    from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
    from torchwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
    from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
    from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
    from .historical_hamiltonian import HistoricalLogDerivative
    from .source_calibration_hamiltonian import (
        SourceCalibrationHamiltonian, SourceCalibratedHistoricalHamiltonian)
    from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
    from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters

    values = [torch.as_tensor(value, dtype=omega_start.dtype, device=omega_start.device)
              for value in (mass1, mass2, chi1, chi2)]
    start, m1, m2, s1, s2 = torch.broadcast_tensors(omega_start, *values)
    if not bool((torch.isfinite(start) & (start > 0)).all()):
        raise ValueError("omega_start must be finite and positive")
    dtype, device = start.dtype, start.device
    batch_shape = tuple(start.shape)
    rows = math.prod(batch_shape) if batch_shape else 1
    maximum_steps = int(max_accepted_steps if max_accepted_steps is not None else num_steps)
    mass_fraction = eob_final_mass_fraction(m1, m2, s1, s2)
    final_spin = hofmann_final_spin(m1, m2, s1, s2, fit_order=final_spin_fit_order)
    qnm_w, qnm_damp = signed_qnm_220_table_convention(final_spin)
    sigma = torch.complex(qnm_w, -qnm_damp) / mass_fraction
    scaled1, scaled2 = m1 / torch.maximum(m1, m2), m2 / torch.maximum(m1, m2)
    x1, x2 = scaled1 / (scaled1 + scaled2), scaled2 / (scaled1 + scaled2)
    nu, chi = seobnrv4_nqc_parameters(x1, x2, s1, s2)
    hamiltonian_type = {
        ("analytic", "published"): SEOBNRv4Hamiltonian,
        ("lal_v4", "published"): HistoricalLogDerivative,
        ("analytic", "lal_v4_rounded"): SourceCalibrationHamiltonian,
        ("lal_v4", "lal_v4_rounded"): SourceCalibratedHistoricalHamiltonian,
    }[(diagnostic_hamiltonian_log_derivative, diagnostic_hamiltonian_calibration)]
    ham = hamiltonian_type(nu, s1, s2, delta=x1 - x2)
    rr = SEOBNRv4PublishedFlux(spin_truncation=spin_truncation,
                               retain_pan_cubic_spin=True,
                               mode_21_higher_spin=diagnostic_mode_21_higher_spin,
                               mode_31_higher_spin=diagnostic_mode_31_higher_spin,
                               mode_32_spin_square=diagnostic_mode_32_spin_square,
                               mode_33_higher_spin=diagnostic_mode_33_higher_spin,
                               mode_43_higher_spin=diagnostic_mode_43_higher_spin)
    # A scalar Hamiltonian is needed for each fine row.  Slicing its tensor
    # parameters preserves the graph while avoiding accidental broadcasting
    # of a row state back across the whole batch.
    flat_nu, flat_s1, flat_s2 = nu.reshape(-1), s1.reshape(-1), s2.reshape(-1)
    flat_delta = (x1 - x2).reshape(-1)
    row_hamiltonians = [hamiltonian_type(
        flat_nu[row], flat_s1[row], flat_s2[row], delta=flat_delta[row])
        for row in range(rows)]
    if diagnostic_warm_start_Momega is None:
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(ham, rr, start)
        initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0), dim=-1)
        initial_flat = initial.reshape(rows, 4)
    else:
        from .warm_start import native_frequency_warm_start
        flat_start = start.reshape(-1)
        initial_rows = []
        for row in range(rows):
            warm_frequency = flat_start[row].new_tensor(diagnostic_warm_start_Momega)
            if bool(flat_start[row] <= warm_frequency):
                r0, pr0, pphi0 = find_quasicircular_initial_conditions(
                    row_hamiltonians[row], rr, flat_start[row])
                initial_rows.append(torch.stack((r0, torch.zeros_like(r0), pr0, pphi0)))
            else:
                result = native_frequency_warm_start(
                    row_hamiltonians[row], rr, warm_frequency, flat_start[row])
                initial_rows.append(result.state)
        initial_flat = torch.stack(initial_rows)
    # Run every coarse row through a scalar closure.  A full-batch EOB field
    # would bind all Hamiltonian parameters while the adaptive loop selected a
    # shrinking subset after row-specific event histories diverged.
    coarse_runs = []
    for row in range(rows):
        row_field = EOBTrajectoryIntegrator(row_hamiltonians[row], rr).vector_field
        coarse_run = integrate_adaptive(
            row_field, initial_flat[row], dt_geom,
            event=AlignedSpinV4Event("coarse"), atol=atol, rtol=rtol,
            initial_step=diagnostic_coarse_initial_step_M,
            controller=GSL_STATE_SCALED_CONTROLLER,
            max_rejected_steps=max_rejected_steps, max_evaluations=max_evaluations,
            max_accepted_steps=maximum_steps, max_duration=max_duration,
        )
        coarse_runs.append(
            _diagnostic_rkf45_output(coarse_run, row_field)
            if diagnostic_rkf45_resampling else
            _diagnostic_hermite_output(coarse_run)
            if diagnostic_hermite_resampling else coarse_run)
    coarse_counts = [int(run.output_valid.sum().detach()) for run in coarse_runs]
    if len(set(coarse_counts)) != 1:
        raise ValueError(
            "adaptive EOB compatibility requires equal coarse output lengths "
            f"across a batch; got {coarse_counts}"
        )
    coarse = (coarse_runs[0] if not batch_shape else
              _stack_adaptive_trajectories(coarse_runs, batch_shape))
    n_coarse = coarse_counts[0]
    if n_coarse < 4:
        raise AdaptiveIntegrationError("short_coarse_trajectory", "coarse trajectory needs at least four output samples")
    duration = (n_coarse - 1) * float(dt_geom)
    if duration >= 150.0:
        restart_index = n_coarse - math.ceil(150.0 / float(dt_geom))
    else:
        restart_index = n_coarse - math.ceil(n_coarse / 2.0)
    if diagnostic_restart_time_M is not None:
        restart_index = round(diagnostic_restart_time_M / dt_geom)
    if restart_index < 1 or restart_index >= n_coarse:
        raise AdaptiveIntegrationError("invalid_restart_index", "two-stage restart index is outside the coarse trajectory")
    coarse_state = coarse.output_state
    coarse_time = coarse.output_time
    coarse_state_flat = coarse_state.reshape(rows, n_coarse, 4)
    coarse_time_flat = coarse_time.reshape(rows, n_coarse)
    restart_state = _preserve_output_phase(coarse_state_flat[:, restart_index])
    restart_time = coarse_time_flat[:, restart_index]
    fine_step = float(dt_geom) / 2.0 ** max(0, math.ceil(math.log2(50.0 * float(dt_geom))))
    fine_events = [AlignedSpinV4Event("fine") for _ in range(rows)]
    fine_runs = []
    for row in range(rows):
        # Calling the same vector field with a single row keeps all parameter
        # tensors on their native graph while making the event count local.
        row_initial = restart_state[row]
        row_field = EOBTrajectoryIntegrator(row_hamiltonians[row], rr).vector_field
        fine_run = integrate_adaptive(
            row_field, row_initial, fine_step, event=fine_events[row],
            atol=atol, rtol=rtol, controller=GSL_STATE_SCALED_CONTROLLER,
            max_rejected_steps=max_rejected_steps, max_evaluations=max_evaluations,
            max_accepted_steps=maximum_steps,
            max_duration=max_duration,
            max_step=diagnostic_fine_max_step_M,
        )
        fine_runs.append(
            _diagnostic_rkf45_output(fine_run, row_field)
            if diagnostic_rkf45_resampling else
            _diagnostic_hermite_output(fine_run)
            if diagnostic_hermite_resampling else fine_run)
    if any(tuple(run.output_state.shape[:-2]) for run in fine_runs):
        raise RuntimeError("unexpected fine trajectory shape")
    fine_counts = [int(run.output_valid.sum().detach()) for run in fine_runs]
    if len(set(fine_counts)) != 1:
        raise ValueError("adaptive EOB compatibility requires equal fine output lengths across a batch")
    n_fine = fine_counts[0]
    refinement = int(round(float(dt_geom) / fine_step))
    if refinement < 1 or refinement & (refinement - 1):
        raise RuntimeError("fine step is not a power-of-two refinement of the coarse step")

    # Build each row's base mode and attachment on the actual fine trajectory.
    attachments = []
    trajectories = []
    from torchwave.approximants.eob.events import OrbitalFrequencyPeak
    for row, run in enumerate(fine_runs):
        row_coarse_state = coarse_state_flat[row, :n_coarse]
        row_fine_state = run.output_state[:n_fine]
        row_ham = row_hamiltonians[row]
        row_field = EOBTrajectoryIntegrator(row_ham, rr).vector_field
        row_coarse_mode, _ = published_mode22_and_frequency(
            row_coarse_state, row_ham, rr, spin_v6=spin_v6,
            include_pan_v8=False, include_positive_chi_540=False,
            retain_pan_cubic_spin=True)
        row_fine_mode, row_fine_frequency = published_mode22_and_frequency(
            row_fine_state, row_ham, rr, spin_v6=spin_v6,
            include_pan_v8=False, include_positive_chi_540=False,
            retain_pan_cubic_spin=True)
        coarse_field = row_field(row_coarse_state)
        fine_field = row_field(row_fine_state)
        # Calibrate on the complete fine series, then carry the same five
        # coefficients back over the coarse prefix.  The raw coarse mode is
        # only an input to this correction; splicing it directly to the
        # corrected fine mode would create an artificial seam.
        row_fine_time = run.output_time[:n_fine]
        # First obtain the continuous peak/match from the actual fine
        # trajectory.  A second attachment evaluation then includes the full
        # finite-support ringdown on its local fine grid before downsampling.
        provisional = attach_seobnrv4_22_numerical_compatibility(
            row_fine_time, row_fine_state, fine_field[..., 1], row_fine_mode,
            row_fine_frequency, nu.reshape(-1)[row], chi.reshape(-1)[row],
            sigma.reshape(-1)[row], row_fine_time,
        )
        match_local = provisional.match_time
        phase_matrix = (_diagnostic_local_nqc_phase_matrix(
            row_fine_time, row_fine_state, fine_field[..., 1], match_local,
            diagnostic_nqc_phase_window_M)
            if diagnostic_local_nqc_derivatives else None)
        matching_second = (_diagnostic_local_matching_second(
            row_fine_time, provisional.nqc, match_local)
            if diagnostic_matching_second_derivatives else None)
        output_count = max(
            2,
            int(torch.ceil((match_local + ringdown_duration).detach() / fine_step).item()) + 1,
        )
        row_output_time = torch.arange(output_count, dtype=dtype, device=device) * fine_step
        attachment = attach_seobnrv4_22_numerical_compatibility(
            row_fine_time, row_fine_state, fine_field[..., 1], row_fine_mode,
            row_fine_frequency, nu.reshape(-1)[row], chi.reshape(-1)[row],
            sigma.reshape(-1)[row], row_output_time,
            diagnostic_phase_matrix=phase_matrix,
            diagnostic_matching_second=matching_second,
        )
        coarse_coefficients = attachment.calibration.coefficients
        row_coarse_mode = _apply_calibrated_nqc(
            row_coarse_mode, row_coarse_state, coarse_field[..., 1],
            coarse_coefficients,
        )
        fine_mode = torch.where(
            row_output_time <= attachment.match_time + ringdown_duration,
            attachment.mode, torch.zeros_like(attachment.mode),
        )
        fine_indices = torch.arange(0, output_count, refinement, device=device)
        actual_indices = fine_indices.clamp_max(n_fine - 1)
        fine_state_samples = row_fine_state[actual_indices]
        fine_frequency_samples = fine_field[actual_indices, 1]
        fine_valid_samples = fine_indices < n_fine
        prefix_mode = row_coarse_mode[:restart_index]
        spliced_mode = torch.cat((prefix_mode, fine_mode[fine_indices]), dim=-1)
        spliced_state = torch.cat((row_coarse_state[:restart_index], fine_state_samples), dim=0)
        spliced_time = torch.cat((coarse_time_flat[row, :restart_index],
                                  restart_time[row] + row_output_time[fine_indices]))
        spliced_frequency = torch.cat((coarse_field[:restart_index, 1], fine_frequency_samples))
        spliced_valid = torch.cat((
            torch.ones((restart_index,), dtype=torch.bool, device=device),
            fine_valid_samples,
        ))
        peak = OrbitalFrequencyPeak(
            attachment.peak.time + restart_time[row], attachment.peak.frequency,
            attachment.peak.curvature, attachment.peak.sample_index)
        attachments.append(SplicedNumericalAttachment(
            spliced_mode, attachment.corrected_inspiral, peak,
            attachment.match_time + restart_time[row],
            attachment.boundary_time + restart_time[row], attachment.boundary_index,
            attachment.boundary_phase, attachment.boundary_frequency,
            attachment.boundary_amplitude, attachment.internal_step,
            attachment, restart_index, refinement))
        trajectories.append((spliced_time, spliced_state, spliced_frequency, peak, spliced_valid))

    if len({item[0].shape[0] for item in trajectories}) != 1:
        raise ValueError("adaptive EOB compatibility requires equal splice lengths across a batch")

    # All rows were checked to have equal lengths; stack while retaining each
    # row's parameter graph.  ``PublishedEOB22`` is imported lazily to keep
    # this module usable by analytic-driver tests without a waveform import.
    from torchwave.approximants.eob.published_waveform import PublishedEOB22
    mode = torch.stack([item.mode for item in attachments]).reshape(batch_shape + (-1,)) if batch_shape else attachments[0].mode
    if batch_shape:
        trajectory_time = torch.stack([item[0] for item in trajectories]).reshape(batch_shape + (-1,))
        trajectory_state = torch.stack([item[1] for item in trajectories]).reshape(batch_shape + (-1, 4))
        trajectory_frequency = torch.stack([item[2] for item in trajectories]).reshape(batch_shape + (-1,))
        trajectory_valid = torch.stack([item[4] for item in trajectories]).reshape(batch_shape + (-1,))
        peak = attachments[0].peak
        # The public FD path accepts a batch peak tensor.  Build it from all
        # rows rather than broadcasting row zero's event metadata.
        peak = OrbitalFrequencyPeak(
            torch.stack([item.peak.time for item in attachments]).reshape(batch_shape),
            torch.stack([item.peak.frequency for item in attachments]).reshape(batch_shape),
            torch.stack([item.peak.curvature for item in attachments]).reshape(batch_shape),
            torch.stack([item.peak.sample_index for item in attachments]).reshape(batch_shape),
        )
        # The public FD adapter needs row-wise match times and mode metadata.
        # Keep the first row's opaque calibration object as an implementation
        # detail while stacking all numerical fields that participate in the
        # public result.
        attachment = SplicedNumericalAttachment(
            mode=mode,
            corrected_inspiral=torch.stack([item.corrected_inspiral for item in attachments]),
            peak=peak,
            match_time=torch.stack([item.match_time for item in attachments]).reshape(batch_shape),
            boundary_time=torch.stack([item.boundary_time for item in attachments]).reshape(batch_shape),
            boundary_index=torch.stack([item.boundary_index for item in attachments]).reshape(batch_shape),
            boundary_phase=torch.stack([item.boundary_phase for item in attachments]).reshape(batch_shape),
            boundary_frequency=torch.stack([item.boundary_frequency for item in attachments]).reshape(batch_shape),
            boundary_amplitude=torch.stack([item.boundary_amplitude for item in attachments]).reshape(batch_shape),
            internal_step=torch.stack([item.internal_step for item in attachments]).reshape(batch_shape),
            fine=attachments[0].fine,
            restart_index=restart_index,
            refinement_factor=refinement,
        )
    else:
        trajectory_time, trajectory_state, trajectory_frequency, peak, trajectory_valid = trajectories[0]
        attachment = attachments[0]
    fine_trajectory = (fine_runs[0] if not batch_shape else
                       _stack_adaptive_trajectories(fine_runs, batch_shape))
    trajectory = TwoStageEOBTrajectory(
        coarse=coarse, fine=fine_trajectory,
        restart_index=restart_index, restart_time=restart_time,
        peak=peak, time=trajectory_time, state=trajectory_state,
        frequency=trajectory_frequency,
        valid=trajectory_valid,
    )
    public_time = trajectory_time if not batch_shape else trajectory_time.reshape(rows, -1)[0]
    return PublishedEOB22(
        public_time, mode,
        trajectory, attachment, mass_fraction, final_spin,
    )


reconstruct_seobnrv4_22_numerical_compatibility = reconstruct_seobnrv4_22_adaptive
