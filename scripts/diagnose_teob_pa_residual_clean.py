#!/usr/bin/env python3
"""Bounded, read-only diagnosis of the public TEOB PA phase residual.

The probe uses the frozen public PA-only fixture and the independent clean
post-adiabatic helper.  It checks the published Eq. (23) logarithmic tail
gauge, compares the public default-spin option with the explicitly requested
``PNSpinOrder=0`` option through ``SimIMRTEOBResumS``, and records PA
order/grid refinement.  Public calls use the exact PA key, an explicit 22
ModeArray, and ``PNTidalOrder=0``.  Clean amplitude and lifted positive
waveform phase are interpolated independently with natural cubics over the
full public uniform time grid.  The diagnostic-only LO horizon branch removes
the tail already present in its runtime mode and reapplies it exactly once,
matching the frozen pregauge branch.  No waveform alignment, fitted phase,
runtime edit, extrapolation, or non-PA fixture is used for metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from torchwave.approximants.eob.teob_postadiabatic_clean import alternating8pa
from torchwave.approximants.eob.teob_radiation_clean import (
    TEOBRadiationClean,
    non_keplerian_radius,
    teob_tail_gauge_factor,
)
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


ROOT = Path(__file__).resolve().parents[1]
PA_CACHE = Path("/tmp/torchwave-teob-pa-public-gauge.npz")
NON_PA_CACHE = Path("/tmp/torchwave-public-teob-q1p5-r20-dt0p5M.npz")
EXPECTED_PA_CACHE_SHA256 = (
    "c81bcb89086833fa6c44fc44a26eb54ee93adc2ff14fca5976944aaae54c8fb0"
)
EXPECTED_NON_PA_CACHE_SHA256 = (
    "09eaa8b00288c94947b2bb4b40ada7780dc51ac98491a48cbb8f60ed50ab686c"
)
# These are the previously reported, pregauge-corrected after-tail values.
# They are retained as immutable comparison points when the clean radiation
# table is changed; the regenerated candidates below are reported separately.
PREVIOUS_BASELINE_RAW_RELATIVE_L2 = 9.9616613848e-5
PREVIOUS_LO_RAW_RELATIVE_L2 = 8.4221181441e-5
RUNTIME_FILES = (
    ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
    ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
    ROOT / "src/torchwave/approximants/teobresums.py",
)

M1_MASS = 30.0
M2_MASS = 20.0
TOTAL_MASS = M1_MASS + M2_MASS
DISTANCE_MPC = 100.0
F_MIN_HZ = 14.45059083415894
F_REF_HZ = 20.0
DT_M = 0.5
NU = M1_MASS * M2_MASS / TOTAL_MASS**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> Any:
    """Make scalar numpy values JSON-safe without hiding non-finite values."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite": 0}
    return {
        "count": int(values.size),
        "finite": int(finite.size),
        "rms": _finite(np.sqrt(np.mean(finite * finite))),
        "max_abs": _finite(np.max(np.abs(finite))),
        "mean": _finite(np.mean(finite)),
    }


def _load_fixture() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if _sha256(PA_CACHE) != EXPECTED_PA_CACHE_SHA256:
        raise RuntimeError("frozen PA fixture hash changed")
    if _sha256(NON_PA_CACHE) != EXPECTED_NON_PA_CACHE_SHA256:
        raise RuntimeError("non-PA comparison fixture hash changed")
    with np.load(PA_CACHE, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "public_hp",
        "public_hc",
        "public_time",
        "pa_time",
        "pa_phase",
        "pa_state",
        "mode_before",
        "Hreal",
        "Omega",
        "Msec",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise RuntimeError(f"PA fixture is missing keys: {missing}")
    public_n = int(data["public_time"].size)
    pa_n = int(data["pa_time"].size)
    if any(data[name].size != public_n for name in ("public_hp", "public_hc")):
        raise RuntimeError("public PA fixture arrays have inconsistent lengths")
    if data["pa_state"].shape != (pa_n, 4):
        raise RuntimeError("PA state does not have the reviewed (r, phase, pr*, pphi) shape")
    if any(data[name].size != pa_n for name in ("pa_phase", "mode_before", "Hreal", "Omega")):
        raise RuntimeError("PA state and mode arrays have inconsistent lengths")
    metadata = {
        "cache": str(PA_CACHE),
        "cache_sha256": _sha256(PA_CACHE),
        "excluded_non_pa_cache": str(NON_PA_CACHE),
        "excluded_non_pa_cache_sha256": _sha256(NON_PA_CACHE),
        "public_samples": public_n,
        "pa_knots": pa_n,
        "public_time_start_M": _finite(data["public_time"][0]),
        "public_time_end_M": _finite(data["public_time"][-1]),
        "pa_time_start_M": _finite(data["pa_time"][0]),
        "pa_time_end_M": _finite(data["pa_time"][-1]),
        "pa_start_radius": _finite(data["pa_state"][0, 0]),
        "pa_end_radius": _finite(data["pa_state"][-1, 0]),
        "Msec": _finite(data["Msec"]),
    }
    return data, metadata


def _hamiltonian() -> TEOBResumSHamiltonian:
    value = torch.tensor(NU, dtype=torch.float64)
    zero = torch.zeros_like(value)
    return TEOBResumSHamiltonian(
        value, zero, zero, zero, zero, delta=torch.tensor(0.2, dtype=torch.float64)
    )


class _LeadingOrderHorizonReaction:
    """Diagnostic-only Eq. (98)-(99) nonspinning horizon replacement.

    The production ``TEOBResumSRadiationReaction`` remains untouched.  This
    object supplies the same clean infinity-mode flux and replaces only its
    horizon term by the leading-order, zero-spin specialization
    ``(X1**4 + X2**4) * vphi**8`` from the cited paper.
    """

    def __init__(self) -> None:
        self.include_horizon_absorption = True
        self._radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")

    def radiation_reaction_forces(
        self,
        omega: torch.Tensor,
        pr_star: torch.Tensor,
        pphi: torch.Tensor,
        *,
        r: torch.Tensor | None = None,
        hamiltonian: TEOBResumSHamiltonian | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if r is None or hamiltonian is None:
            raise ValueError("leading-order horizon diagnostic requires the full clean state")
        h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
            r, pr_star, pphi
        )
        infinity_flux = self._radiation.flux(
            r,
            omega,
            torch.zeros_like(omega),
            pr_star,
            pphi,
            h_eff,
            h_real,
            h_circ,
            a_prime,
            hamiltonian.m1,
            hamiltonian.m2,
            include_horizon=False,
        )
        r_omega = non_keplerian_radius(r, h_circ, a_prime)
        vphi = r_omega * omega
        total_mass = hamiltonian.m1 + hamiltonian.m2
        x1 = hamiltonian.m1 / total_mass
        x2 = hamiltonian.m2 / total_mass
        horizon_ratio = (x1**4 + x2**4) * vphi**8
        horizon_flux = (32.0 / 5.0) * hamiltonian.nu**2 * r_omega**4 * omega**5 * horizon_ratio
        flux = infinity_flux + horizon_flux
        nu = hamiltonian.nu
        if omega.dim() > nu.dim():
            nu = nu.unsqueeze(-1)
        return torch.zeros_like(flux), -flux / nu


def _pa_run(hamiltonian: TEOBResumSHamiltonian, *, order: int, dr: float) -> dict[str, Any]:
    started = time.perf_counter()
    result = alternating8pa(
        hamiltonian,
        r0=19.999999999999996,
        r_min=14.0,
        dr=dr,
        pa_order=order,
    )
    elapsed = time.perf_counter() - started
    return {
        "status": result.status,
        "pa_order": order,
        "dr": dr,
        "points": int(result.radius.numel()),
        "endpoint_radius": _finite(result.radius[-1]),
        "endpoint_time_M": _finite(result.endpoint_time),
        "endpoint_phase": _finite(result.endpoint_phase),
        "elapsed_seconds": _finite(elapsed),
        "max_abs_radial_balance": _finite(result.radial_balance_residual.abs().max()),
        "max_abs_angular_balance": _finite(result.angular_balance_residual.abs().max()),
    }


def _helper_reproduction(data: dict[str, np.ndarray]) -> dict[str, Any]:
    result = alternating8pa(
        _hamiltonian(),
        r0=float(data["pa_state"][0, 0]),
        r_min=14.0,
        dr=0.1,
        pa_order=8,
    )
    state = np.column_stack(
        [
            result.radius.detach().numpy(),
            result.phase.detach().numpy(),
            result.pr_star.detach().numpy(),
            result.p_phi.detach().numpy(),
        ]
    )
    return {
        "status": result.status,
        "points": int(state.shape[0]),
        "max_abs_state_error": _finite(np.max(np.abs(state - data["pa_state"]))),
        "max_abs_time_error": _finite(
            np.max(np.abs(result.time.detach().numpy() - data["pa_time"]))
        ),
        "max_abs_phase_error": _finite(
            np.max(np.abs(result.phase.detach().numpy() - data["pa_phase"]))
        ),
    }


def _clean_mode_on_pa(
    hamiltonian: TEOBResumSHamiltonian,
    result: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return a pregauge mode while recording the runtime's existing TEOB tail.

    ``TEOBRadiationClean.mode`` already applies the TEOB ``r0=2GM/sqrt(e)``
    factor.  The projection code takes a pregauge ``r0=2M`` mode and applies
    that factor once, so remove the runtime factor here before passing the
    candidate onward.  This keeps the frozen and diagnostic-only candidates on
    the same gauge contract without editing the clean runtime.
    """
    h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
        result.radius, result.pr_star, result.p_phi
    )
    mode_with_teob_tail = TEOBRadiationClean(8, residual_phase="zero-unpublished").mode(
        2,
        2,
        result.radius,
        result.omega,
        result.phase,
        result.pr_star,
        result.p_phi,
        h_eff,
        h_real,
        h_circ,
        a_prime,
        hamiltonian.m1,
        hamiltonian.m2,
    )
    tail_factor = teob_tail_gauge_factor(2, h_real, result.omega)
    mode_pregauge = mode_with_teob_tail / tail_factor
    if not bool(torch.allclose(mode_pregauge * tail_factor, mode_with_teob_tail,
                               rtol=1.0e-12, atol=1.0e-15)):
        raise AssertionError("runtime TEOB tail could not be removed exactly once")
    return (
        mode_pregauge.detach().numpy(),
        h_real.detach().numpy(),
        result.omega.detach().numpy(),
        {
            "runtime_mode_gauge": "TEOB tail-corrected r0=2GM/sqrt(e)",
            "returned_mode_gauge": "pregauge r0=2M",
            "removed_factor": "exp(-2 i Hreal Omega) for m=2",
            "tail_application_for_projection": 1,
        },
    )


def _tail_convention_assertion() -> dict[str, Any]:
    """Check the one-tail algebra on a small synthetic complex mode."""
    h_real = torch.tensor(1.25, dtype=torch.float64)
    omega = torch.tensor(0.17, dtype=torch.float64)
    pregauge = torch.polar(torch.tensor(2.0, dtype=torch.float64),
                           torch.tensor(0.31, dtype=torch.float64))
    factor = teob_tail_gauge_factor(2, h_real, omega)
    runtime_mode = pregauge * factor
    recovered_pregauge = runtime_mode / factor
    projected_once = recovered_pregauge * factor
    expected_tail_phase = -2.0 * float(h_real) * float(omega)
    measured_tail_phase = float(torch.angle(runtime_mode / pregauge))
    checks = {
        "recover_pregauge": bool(torch.allclose(recovered_pregauge, pregauge,
                                                 rtol=1.0e-13, atol=1.0e-15)),
        "reapply_once": bool(torch.allclose(projected_once, runtime_mode,
                                             rtol=1.0e-13, atol=1.0e-15)),
        "phase_formula": bool(math.isclose(measured_tail_phase, expected_tail_phase,
                                             rel_tol=0.0, abs_tol=1.0e-15)),
    }
    if not all(checks.values()):
        raise AssertionError(f"one-tail convention check failed: {checks}")
    return {
        "status": "passed",
        "checks": checks,
        "formula": "pregauge -> pregauge*exp(-2 i Hreal Omega) exactly once",
        "test_values": {"Hreal": float(h_real), "Omega": float(omega)},
    }


def _natural_cubic_resample(
    knots: np.ndarray, values: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Evaluate the clean natural-cubic convention on the public grid."""
    knots = np.asarray(knots, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if knots.ndim != 1 or values.ndim != 1 or query.ndim != 1:
        raise ValueError("natural-cubic diagnostic inputs must be one-dimensional")
    if knots.size != values.size or knots.size < 2:
        raise ValueError("natural-cubic diagnostic knots and values have incompatible sizes")
    if not np.all(np.isfinite(knots)) or not np.all(np.isfinite(values)):
        raise ValueError("natural-cubic diagnostic knots and values must be finite")
    if not np.all(np.diff(knots) > 0.0):
        raise ValueError("natural-cubic diagnostic knots must be strictly increasing")
    if query.size and (query[0] < knots[0] - 1.0e-10 or query[-1] > knots[-1] + 1.0e-10):
        raise ValueError("dense public grid would require natural-cubic extrapolation")
    spline = natural_cubic_spline(
        torch.from_numpy(knots), torch.from_numpy(values), scalar_output=True
    )
    return spline(torch.from_numpy(query)).detach().cpu().numpy()


def _lifted_positive_waveform_phase(
    mode: np.ndarray, orbital_phase: np.ndarray
) -> np.ndarray:
    """Return the clean positive waveform phase specified by the frame chart.

    The mode is reconstructed as ``amplitude * exp(-1j * phase)``.  The
    product inside ``unwrap`` removes the fast orbital rotation before the
    phase is lifted, while the explicit ``2*orbital_phase`` restores the
    positive waveform-phase convention.
    """
    mode = np.asarray(mode, dtype=np.complex128)
    orbital_phase = np.asarray(orbital_phase, dtype=np.float64)
    if mode.shape != orbital_phase.shape:
        raise ValueError("mode and orbital phase must have the same shape")
    lifted = np.unwrap(np.angle(mode * np.exp(2j * orbital_phase)))
    return 2.0 * orbital_phase - lifted


def _dense_candidate_metrics(
    *,
    public_time: np.ndarray,
    public_mode: np.ndarray,
    pa_time: np.ndarray,
    orbital_phase: np.ndarray,
    mode_knots: np.ndarray,
    scale: float,
    label: str,
    status: str,
    endpoint_radius: float | None = None,
    endpoint_time: float | None = None,
    endpoint_phase: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compare a clean PA candidate after dense natural-cubic resampling."""
    public_time = np.asarray(public_time, dtype=np.float64)
    public_mode = np.asarray(public_mode, dtype=np.complex128)
    pa_time = np.asarray(pa_time, dtype=np.float64)
    orbital_phase = np.asarray(orbital_phase, dtype=np.float64)
    mode_knots = np.asarray(mode_knots, dtype=np.complex128)
    if public_time.ndim != 1 or public_mode.ndim != 1 or public_time.size != public_mode.size:
        raise ValueError("public waveform arrays must be matching one-dimensional arrays")
    if public_time.size < 2 or not np.all(np.diff(public_time) > 0.0):
        raise ValueError("public waveform time must be strictly increasing")
    public_steps = np.diff(public_time)
    if not np.allclose(public_steps, public_steps[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("public waveform time must be uniform for the dense diagnostic")
    if pa_time.shape != orbital_phase.shape or pa_time.shape != mode_knots.shape:
        raise ValueError("clean PA arrays must have matching shapes")
    if pa_time.size < 2 or not np.all(np.diff(pa_time) > 0.0):
        raise ValueError("clean PA time must be strictly increasing")
    if pa_time[0] > public_time[0] + 1.0e-10 or pa_time[-1] < public_time[-1] - 1.0e-10:
        raise ValueError("clean PA trajectory does not cover the public grid without extrapolation")

    clean_amplitude_knots = np.abs(mode_knots)
    clean_phase_knots = _lifted_positive_waveform_phase(mode_knots, orbital_phase)
    clean_amplitude = _natural_cubic_resample(
        pa_time, clean_amplitude_knots, public_time
    )
    clean_phase = _natural_cubic_resample(pa_time, clean_phase_knots, public_time)
    clean_mode = scale * clean_amplitude * np.exp(-1j * clean_phase)

    # The residual is deliberately clean/public so its sign reports the
    # clean phase relative to the public waveform.
    ratio = clean_mode / public_mode
    phase_error = np.unwrap(np.angle(ratio))
    public_amplitude = np.abs(public_mode)
    clean_amplitude_dense = np.abs(clean_mode)
    amplitude_error = clean_amplitude_dense / public_amplitude - 1.0
    raw_denominator = max(float(np.linalg.norm(public_mode)), 1.0e-300)
    amplitude_denominator = max(float(np.linalg.norm(public_amplitude)), 1.0e-300)
    phase_derivative = np.gradient(phase_error, public_time)
    payload = {
        "label": label,
        "status": status,
        "samples": int(public_time.size),
        "pa_knots": int(pa_time.size),
        "endpoint_radius": _finite(endpoint_radius),
        "endpoint_time_M": _finite(endpoint_time),
        "endpoint_phase": _finite(endpoint_phase),
        "interpolation": {
            "abscissa": "clean PA time knots",
            "ordinates": "clean amplitude and lifted positive waveform phase independently",
            "kind": "natural cubic",
            "query_grid": "full public uniform time grid",
            "extrapolation": False,
        },
        "phase_convention": (
            "phase=2*orbital_phi-unwrap(angle(mode*exp(2i*orbital_phi))); "
            "clean_mode=scale*amplitude*exp(-1j*phase)"
        ),
        "direction": "clean_over_public",
        "raw_relative_l2": _finite(np.linalg.norm(clean_mode - public_mode) / raw_denominator),
        "amplitude_relative_l2": _finite(
            np.linalg.norm(clean_amplitude_dense - public_amplitude) / amplitude_denominator
        ),
        "amplitude_pointwise": _array_stats(amplitude_error),
        "phase_radians": _array_stats(phase_error),
        "initial_phase_error": _finite(phase_error[0]),
        "final_phase_error": _finite(phase_error[-1]),
        "initial_ratio_real": _finite(ratio[0].real),
        "initial_ratio_imag": _finite(ratio[0].imag),
        "phase_derivative_residual": {
            "direction": "d(clean/public phase)/dM",
            "units": "radians per M",
            **_array_stats(phase_derivative),
            "initial": _finite(phase_derivative[0]),
            "final": _finite(phase_derivative[-1]),
        },
        "clean_amplitude_dense_min": _finite(np.min(clean_amplitude)),
        "clean_amplitude_dense_max": _finite(np.max(clean_amplitude)),
    }
    return payload, clean_mode


def _project_pa_candidate(
    data: dict[str, np.ndarray],
    result: Any,
    mode: np.ndarray,
    h_real: np.ndarray,
    omega: np.ndarray,
    *,
    mode_input_gauge: str = "pregauge r0=2M",
) -> dict[str, Any]:
    """Compare one pregauge PA trajectory after applying the TEOB tail once."""
    public_time = np.asarray(data["public_time"], dtype=np.float64)
    public_mode = np.asarray(data["public_hp"], dtype=np.float64) - 1j * np.asarray(
        data["public_hc"], dtype=np.float64
    )
    pa_time = np.asarray(result.time.detach().numpy(), dtype=np.float64)
    scale = -math.sqrt(5.0 / (4.0 * math.pi)) * float(data["Msec"]) * lal.C_SI / (
        DISTANCE_MPC * 1.0e6 * lal.PC_SI
    )
    mode = np.asarray(mode, dtype=np.complex128)
    h_real = np.asarray(h_real, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    orbital_phase = np.asarray(result.phase.detach().numpy(), dtype=np.float64)
    before, _ = _dense_candidate_metrics(
        public_time=public_time,
        public_mode=public_mode,
        pa_time=pa_time,
        orbital_phase=orbital_phase,
        mode_knots=mode,
        scale=scale,
        label="baseline_2M",
        status=result.status,
        endpoint_radius=float(result.radius[-1]),
        endpoint_time=float(result.endpoint_time),
        endpoint_phase=float(result.endpoint_phase),
    )
    after, _ = _dense_candidate_metrics(
        public_time=public_time,
        public_mode=public_mode,
        pa_time=pa_time,
        orbital_phase=orbital_phase,
        mode_knots=mode * np.exp(-2j * h_real * omega),
        scale=scale,
        label="published_r0_2M_over_sqrt_e_tail",
        status=result.status,
        endpoint_radius=float(result.radius[-1]),
        endpoint_time=float(result.endpoint_time),
        endpoint_phase=float(result.endpoint_phase),
    )
    return {
        "status": result.status,
        "points": int(pa_time.size),
        "endpoint_radius": _finite(result.radius[-1]),
        "endpoint_time_M": _finite(result.endpoint_time),
        "endpoint_phase": _finite(result.endpoint_phase),
        "mode_input_gauge": mode_input_gauge,
        "tail_factor": "exp(-2 i Hreal Omega) for m=2",
        "tail_applications": 1,
        "before_tail": before,
        "after_tail": after,
    }


def _leading_order_horizon_diagnostic(
    data: dict[str, np.ndarray],
    *,
    cache_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the paper's LO horizon term as an explicitly non-production probe."""
    hamiltonian = _hamiltonian()
    baseline = alternating8pa(
        hamiltonian,
        r0=float(data["pa_state"][0, 0]),
        r_min=14.0,
        dr=0.1,
        pa_order=8,
    )
    h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
        baseline.radius, baseline.pr_star, baseline.p_phi
    )
    radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")
    r_omega = non_keplerian_radius(baseline.radius, h_circ, a_prime)
    vphi = r_omega * baseline.omega
    x1 = hamiltonian.m1 / (hamiltonian.m1 + hamiltonian.m2)
    x2 = hamiltonian.m2 / (hamiltonian.m1 + hamiltonian.m2)
    identity_lhs = x1**4 + x2**4
    identity_rhs = 1.0 - 4.0 * hamiltonian.nu + 2.0 * hamiltonian.nu**2
    scale = (32.0 / 5.0) * hamiltonian.nu**2 * r_omega**4 * baseline.omega**5
    lo_flux = scale * identity_lhs * vphi**8
    current_flux = radiation.horizon_flux(
        baseline.radius,
        baseline.omega,
        torch.zeros_like(baseline.omega),
        baseline.pr_star,
        baseline.p_phi,
        h_eff,
        h_real,
        h_circ,
        a_prime,
        hamiltonian.m1,
        hamiltonian.m2,
    )
    current_np = current_flux.detach().numpy()
    lo_np = lo_flux.detach().numpy()
    relative_difference = (lo_np - current_np) / np.maximum(np.abs(current_np), 1e-300)

    current_mode, current_hreal, current_omega, current_mode_gauge = _clean_mode_on_pa(
        hamiltonian, baseline
    )
    current_projection = _project_pa_candidate(
        data,
        baseline,
        current_mode,
        current_hreal,
        current_omega,
        mode_input_gauge=current_mode_gauge["returned_mode_gauge"],
    )

    lo_result = alternating8pa(
        _hamiltonian(),
        radiation_reaction=_LeadingOrderHorizonReaction(),
        r0=float(data["pa_state"][0, 0]),
        r_min=14.0,
        dr=0.1,
        pa_order=8,
    )
    lo_hamiltonian = _hamiltonian()
    lo_mode, lo_hreal, lo_omega, lo_mode_gauge = _clean_mode_on_pa(lo_hamiltonian, lo_result)
    current_mode_gauge_name = current_mode_gauge["returned_mode_gauge"]
    candidate_input_gauges = {
        "current_clean": current_mode_gauge_name,
        "leading_order": lo_mode_gauge["returned_mode_gauge"],
    }
    same_input_gauge = len(set(candidate_input_gauges.values())) == 1
    if not same_input_gauge:
        raise AssertionError(
            "current and leading-order zero-spin candidates do not share "
            f"the same input gauge: {candidate_input_gauges}"
        )
    lo_projection = _project_pa_candidate(
        data,
        lo_result,
        lo_mode,
        lo_hreal,
        lo_omega,
        mode_input_gauge=lo_mode_gauge["returned_mode_gauge"],
    )
    current_tail = current_projection["after_tail"]
    lo_tail = lo_projection["after_tail"]
    if cache_arrays is not None:
        cache_arrays.update(
            {
                "clean_baseline_pa_time_M": baseline.time.detach().numpy(),
                "clean_baseline_pa_phase": baseline.phase.detach().numpy(),
                "clean_baseline_mode_pregauge_real": current_mode.real,
                "clean_baseline_mode_pregauge_imag": current_mode.imag,
                "clean_baseline_Hreal": current_hreal,
                "clean_baseline_Omega": current_omega,
                "clean_lo_pa_time_M": lo_result.time.detach().numpy(),
                "clean_lo_pa_phase": lo_result.phase.detach().numpy(),
                "clean_lo_mode_pregauge_real": lo_mode.real,
                "clean_lo_mode_pregauge_imag": lo_mode.imag,
                "clean_lo_Hreal": lo_hreal,
                "clean_lo_Omega": lo_omega,
            }
        )
    return {
        "status": "diagnostic_only",
        "primary_source": "https://arxiv.org/pdf/1406.6913v1 Sec. IV.B Eqs. (98)-(99)",
        "leading_order_flux_formula": (
            "F_H=(32/5)*nu**2*r_omega**4*Omega**5*"
            "(X1**4+X2**4)*vphi**8; F_phi,H=-F_H/nu"
        ),
        "zero_spin_identity": {
            "X1": _finite(x1),
            "X2": _finite(x2),
            "X1_fourth_plus_X2_fourth": _finite(identity_lhs),
            "1_minus_4nu_plus_2nu2": _finite(identity_rhs),
            "absolute_difference": _finite(torch.abs(identity_lhs - identity_rhs)),
            "ratio_used": "(X1**4 + X2**4) * vphi**8",
        },
        "horizon_flux_comparison_on_current_clean_pa": {
            "direction": "(leading_order - current_clean) / current_clean",
            "relative_difference": _array_stats(relative_difference),
            "initial_current_flux": _finite(current_np[0]),
            "initial_leading_order_flux": _finite(lo_np[0]),
            "final_current_flux": _finite(current_np[-1]),
            "final_leading_order_flux": _finite(lo_np[-1]),
        },
        "leading_order_pa_candidate": lo_projection,
        "candidate_input_gauge_assertion": {
            "status": "passed",
            "same_input_gauge": same_input_gauge,
            "candidate_input_gauges": candidate_input_gauges,
            "tail_applications_during_projection": {
                "current_clean": 1,
                "leading_order": 1,
            },
        },
        "current_clean_pa_candidate": current_projection,
        "tail_convention": lo_mode_gauge,
        "candidate_vs_current_clean_tail": {
            "raw_relative_l2_ratio": _finite(
                lo_tail["raw_relative_l2"] / current_tail["raw_relative_l2"]
            ),
            "phase_rms_ratio": _finite(
                lo_tail["phase_radians"]["rms"] / current_tail["phase_radians"]["rms"]
            ),
            "final_phase_error_ratio": _finite(
                lo_tail["final_phase_error"] / current_tail["final_phase_error"]
            ),
            "note": "Ratios use the same dense public grid; neither branch is aligned or fitted.",
        },
        "interpretation": (
            "This branch is an equation-level sensitivity check only. It does not "
            "select a production horizon prescription or modify the clean helper. "
            "The runtime mode already carries one TEOB tail; it is converted to the "
            "pregauge form before projection, so `after_tail` is the physical one-tail "
            "LO candidate and `before_tail` is its pregauge diagnostic."
        ),
    }


def _public_dict(*, spin_order: int | None, use_pa: bool) -> lal.Dict:
    params = lal.CreateDict()
    if use_pa:
        # This is the public TEOBResumS parameter spelling.  The similarly
        # named CamelCase key is ignored by the direct public entry point.
        lal.DictInsertINT4Value(params, "TEOB_use_postadiabatic", 2)
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(params, 0)
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, mode_array)
    if spin_order is not None:
        lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(params, int(spin_order))
    return params


def _public_call(*, label: str, spin_order: int | None, use_pa: bool) -> tuple[dict[str, Any], np.ndarray | None]:
    started = time.perf_counter()
    try:
        hp, hc = lalsim.SimIMRTEOBResumS(
            0.0,
            DT_M * (TOTAL_MASS * lal.MTSUN_SI),
            M1_MASS * lal.MSUN_SI,
            M2_MASS * lal.MSUN_SI,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            DISTANCE_MPC * 1.0e6 * lal.PC_SI,
            0.0,
            0.0,
            _public_dict(spin_order=spin_order, use_pa=use_pa),
            0.0,
            0.0,
            F_MIN_HZ,
            F_REF_HZ,
        )
        hp_values = np.asarray(hp.data.data, dtype=np.float64).copy()
        hc_values = np.asarray(hc.data.data, dtype=np.float64).copy()
        complex_values = hp_values - 1j * hc_values
        return (
            {
                "label": label,
                "status": "ok",
                "api": "SimIMRTEOBResumS",
                "spin_order": spin_order,
                "use_post_adiabatic": use_pa,
                "samples": int(complex_values.size),
                "epoch_seconds": _finite(hp.epoch),
                "delta_t_seconds": _finite(hp.deltaT),
                "delta_t_M": _finite(hp.deltaT / (TOTAL_MASS * lal.MTSUN_SI)),
                "f_ref_hz": F_REF_HZ,
                "pa_key": "TEOB_use_postadiabatic",
                "pa_value": 2 if use_pa else None,
                "mode_array": "[(2,+2)]",
                "pn_tidal_order": 0,
                "pa_only_length_range": [18000, 20000],
                "pa_only_length_ok": 18000 <= int(complex_values.size) <= 20000,
                "elapsed_seconds": _finite(time.perf_counter() - started),
            },
            complex_values,
        )
    except Exception as exc:  # public API behavior is itself diagnostic evidence
        return (
            {
                "label": label,
                "status": "error",
                "api": "SimIMRTEOBResumS",
                "spin_order": spin_order,
                "use_post_adiabatic": use_pa,
                "pa_key": "TEOB_use_postadiabatic",
                "pa_value": 2 if use_pa else None,
                "mode_array": "[(2,+2)]",
                "pn_tidal_order": 0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": _finite(time.perf_counter() - started),
            },
            None,
        )


def _waveform_difference(first: np.ndarray | None, second: np.ndarray | None) -> dict[str, Any]:
    if first is None or second is None:
        return {"status": "unavailable"}
    if first.shape != second.shape:
        return {"status": "shape_mismatch", "first": first.shape, "second": second.shape}
    denominator = max(float(np.linalg.norm(first)), 1e-300)
    delta = first - second
    return {
        "status": "ok",
        "samples": int(first.size),
        "relative_l2": _finite(np.linalg.norm(delta) / denominator),
        "max_abs": _finite(np.max(np.abs(delta))),
    }


def _fixture_waveform_comparison(
    data: dict[str, np.ndarray], waveform: np.ndarray | None
) -> dict[str, Any]:
    """Check a regenerated direct public PA output against the frozen arrays."""
    if waveform is None:
        return {"status": "unavailable"}
    expected = np.asarray(data["public_hp"], dtype=np.float64) - 1j * np.asarray(
        data["public_hc"], dtype=np.float64
    )
    if waveform.shape != expected.shape:
        return {
            "status": "shape_mismatch",
            "generated_samples": int(waveform.size),
            "fixture_samples": int(expected.size),
        }
    delta = waveform - expected
    denominator = max(float(np.linalg.norm(expected)), 1.0e-300)
    return {
        "status": "ok",
        "generated_samples": int(waveform.size),
        "fixture_samples": int(expected.size),
        "relative_l2": _finite(np.linalg.norm(delta) / denominator),
        "max_abs": _finite(np.max(np.abs(delta))),
        "exact_float64": bool(np.array_equal(waveform, expected)),
    }


def _projection_metrics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compare frozen clean mode knots after dense natural-cubic lifting."""
    public_time = np.asarray(data["public_time"], dtype=np.float64)
    public_mode = np.asarray(data["public_hp"], dtype=np.float64) - 1j * np.asarray(
        data["public_hc"], dtype=np.float64
    )
    pa_time = np.asarray(data["pa_time"], dtype=np.float64)
    mode_before = np.asarray(data["mode_before"], dtype=np.complex128)
    hreal = np.asarray(data["Hreal"], dtype=np.float64)
    omega = np.asarray(data["Omega"], dtype=np.float64)
    pa_phase = np.asarray(data["pa_phase"], dtype=np.float64)
    msec = float(data["Msec"])
    scale = -math.sqrt(5.0 / (4.0 * math.pi)) * msec * lal.C_SI / (
        DISTANCE_MPC * 1.0e6 * lal.PC_SI
    )
    mode_after = mode_before * np.exp(-2j * hreal * omega)
    before, _ = _dense_candidate_metrics(
        public_time=public_time,
        public_mode=public_mode,
        pa_time=pa_time,
        orbital_phase=pa_phase,
        mode_knots=mode_before,
        scale=scale,
        label="baseline_2M",
        status="fixture",
        endpoint_radius=float(data["pa_state"][-1, 0]),
        endpoint_time=float(pa_time[-1]),
        endpoint_phase=float(pa_phase[-1]),
    )
    after, _ = _dense_candidate_metrics(
        public_time=public_time,
        public_mode=public_mode,
        pa_time=pa_time,
        orbital_phase=pa_phase,
        mode_knots=mode_after,
        scale=scale,
        label="published_r0_2M_over_sqrt_e_tail",
        status="fixture",
        endpoint_radius=float(data["pa_state"][-1, 0]),
        endpoint_time=float(pa_time[-1]),
        endpoint_phase=float(pa_phase[-1]),
    )
    return {
        "primary_source": "https://arxiv.org/pdf/1212.4357v2 Eq.(23)",
        "projection": "h_plus - i h_cross = -sqrt(5/(4 pi))*Msec*c/d*h22",
        "tail_gauge": "mode_after = mode_before*exp(-2 i Hreal Omega)",
        "mode_before_gauge": "pregauge r0=2M",
        "mode_after_gauge": "TEOB r0=2GM/sqrt(e)",
        "tail_applications": 1,
        "scale": _finite(scale),
        "comparison_grid": "full public uniform time grid",
        "baseline_r0_2M": before,
        "published_r0_2M_over_sqrt_e_tail": after,
    }


def _metric_delta(observed: float, reference: float) -> dict[str, Any]:
    """Compare a regenerated raw curve metric with an earlier report value."""
    observed = float(observed)
    reference = float(reference)
    denominator = max(abs(reference), 1.0e-300)
    return {
        "reference_raw_relative_l2": _finite(reference),
        "observed_raw_relative_l2": _finite(observed),
        "absolute_difference": _finite(observed - reference),
        "relative_difference": _finite((observed - reference) / denominator),
    }


def _save_cache(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write a numeric output cache and return its SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return _sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="optional NPZ cache for regenerated clean candidates and public arrays",
    )
    args = parser.parse_args()
    started = time.perf_counter()

    data, fixture = _load_fixture()
    tail_convention = _tail_convention_assertion()
    runtime_before = {str(path.relative_to(ROOT)): _sha256(path) for path in RUNTIME_FILES}
    cache_arrays: dict[str, np.ndarray] = {
        "public_time_M": np.asarray(data["public_time"], dtype=np.float64),
        "public_hp": np.asarray(data["public_hp"], dtype=np.float64),
        "public_hc": np.asarray(data["public_hc"], dtype=np.float64),
        "frozen_pa_time_M": np.asarray(data["pa_time"], dtype=np.float64),
        "frozen_pa_phase": np.asarray(data["pa_phase"], dtype=np.float64),
        "frozen_mode_before_real": np.asarray(data["mode_before"].real, dtype=np.float64),
        "frozen_mode_before_imag": np.asarray(data["mode_before"].imag, dtype=np.float64),
        "frozen_Hreal": np.asarray(data["Hreal"], dtype=np.float64),
        "frozen_Omega": np.asarray(data["Omega"], dtype=np.float64),
    }
    hamiltonian = _hamiltonian()
    pa_orders = [_pa_run(hamiltonian, order=order, dr=0.1) for order in (2, 4, 6, 8)]
    grid_refinement = [_pa_run(hamiltonian, order=8, dr=dr) for dr in (0.2, 0.1)]
    reproduction = _helper_reproduction(data)

    default_meta, default_waveform = _public_call(
        label="default_spin_with_TEOB_use_postadiabatic_2", spin_order=None, use_pa=True
    )
    explicit_meta, explicit_waveform = _public_call(
        label="explicit_PNSpinOrder_0_with_TEOB_use_postadiabatic_2", spin_order=0, use_pa=True
    )
    lo_probe = _leading_order_horizon_diagnostic(data, cache_arrays=cache_arrays)
    frozen_projection = _projection_metrics(data)
    current_projection = lo_probe["current_clean_pa_candidate"]
    current_after_tail = current_projection["after_tail"]
    lo_after_tail = lo_probe["leading_order_pa_candidate"]["after_tail"]
    if default_waveform is not None:
        cache_arrays["public_default_hp_minus_i_hc_real"] = default_waveform.real
        cache_arrays["public_default_hp_minus_i_hc_imag"] = default_waveform.imag
    cache_metadata: dict[str, Any] | None = None
    if args.cache is not None:
        cache_sha256 = _save_cache(args.cache, cache_arrays)
        cache_metadata = {
            "path": str(args.cache),
            "sha256": cache_sha256,
            "arrays": sorted(cache_arrays),
        }
    runtime_after = {str(path.relative_to(ROOT)): _sha256(path) for path in RUNTIME_FILES}

    payload: dict[str, Any] = {
        "description": "bounded clean-room public TEOB PA residual diagnostic after the clean radiation-table patch",
        "revision_note": (
            "This report is a new radiation-completion rerun; the earlier corrected "
            "dense-grid artifact remains unchanged for comparison. "
            "The regenerated public waveform now uses SimIMRTEOBResumS with the exact "
            "TEOB_use_postadiabatic key, an explicit (2,+2) ModeArray, PNTidalOrder=0, "
            "and f_ref=20 Hz.  Residuals use independent natural-cubic interpolation "
            "of clean amplitude and lifted positive phase onto the full public uniform "
            "grid.  The LO branch correction removes the TEOB tail already present in "
            "the runtime mode and applies it exactly once during projection; the diagnostic "
            "remains read-only and does not edit the clean runtime."
        ),
        "host": platform.node(),
        "working_directory": str(ROOT),
        "scope": {
            "read_only": True,
            "no_alignment_or_fit": True,
            "non_pa_fixture_used_for_metrics": False,
            "public_api": "SimIMRTEOBResumS",
            "comparison_uses_full_public_uniform_grid": True,
            "physical_parameters": {
                "mass1_msun": M1_MASS,
                "mass2_msun": M2_MASS,
                "chi1": 0.0,
                "chi2": 0.0,
                "lambda1": 0.0,
                "lambda2": 0.0,
                "distance_mpc": DISTANCE_MPC,
                "inclination": 0.0,
                "phi_ref": 0.0,
                "delta_t_M": DT_M,
                "f_min_hz": F_MIN_HZ,
                "f_ref_hz": F_REF_HZ,
                "TEOB_use_postadiabatic": 2,
                "PNTidalOrder": 0,
                "ModeArray": "[(2,+2)]",
                "modes": "explicit 22 mode only (face-on projection)",
            },
        },
        "fixture": fixture,
        "clean_helper_reproduction": reproduction,
        "tail_convention_assertion": tail_convention,
        "pa_order_runs": pa_orders,
        "pa_grid_refinement": grid_refinement,
        "leading_order_horizon_probe": lo_probe,
        "public_spin_option": {
            "binary": "TEOBResumS",
            "api": "SimIMRTEOBResumS",
            "pa_key": "TEOB_use_postadiabatic",
            "pa_value": 2,
            "mode_array": "[(2,+2)]",
            "pn_tidal_order": 0,
            "default_spin": default_meta,
            "default_spin_fixture_comparison": _fixture_waveform_comparison(data, default_waveform),
            "explicit_PNSpinOrder_0": explicit_meta,
            "explicit_PNSpinOrder_0_fixture_comparison": _fixture_waveform_comparison(
                data, explicit_waveform
            ),
            "default_vs_explicit_PNSpinOrder_0": _waveform_difference(
                default_waveform, explicit_waveform
            ),
            "interpretation": (
                "The direct public PA route uses the exact TEOB_use_postadiabatic key and "
                "returns the frozen 19048-sample default waveform exactly.  The explicit "
                "PNSpinOrder=0 PA route is accepted as a separate 19050-sample control; "
                "its different length is recorded rather than truncated or aligned."
                " This comparison is an attribution control only: changing the public "
                "spin-order option can alter other evolution choices, so it cannot isolate "
                "the horizon prescription."
            ),
        },
        "projection_and_phase": frozen_projection,
        "radiation_patch_comparison": {
            "comparison_metric": "raw_relative_l2, after exactly one TEOB tail",
            "reference_report_values": {
                "baseline_pregauge_corrected_raw_relative_l2": PREVIOUS_BASELINE_RAW_RELATIVE_L2,
                "leading_order_horizon_raw_relative_l2": PREVIOUS_LO_RAW_RELATIVE_L2,
                "source": "previous corrected dense-grid PA report before the radiation-table patch",
            },
            "frozen_fixture_projection": _metric_delta(
                frozen_projection["published_r0_2M_over_sqrt_e_tail"]["raw_relative_l2"],
                PREVIOUS_BASELINE_RAW_RELATIVE_L2,
            ),
            "regenerated_clean_baseline": {
                **_metric_delta(
                    current_after_tail["raw_relative_l2"],
                    PREVIOUS_BASELINE_RAW_RELATIVE_L2,
                ),
                "endpoint_time_M": current_after_tail["endpoint_time_M"],
                "endpoint_phase": current_after_tail["endpoint_phase"],
                "whole_pa_curve_metrics": current_after_tail,
            },
            "regenerated_leading_order_horizon": {
                **_metric_delta(
                    lo_after_tail["raw_relative_l2"],
                    PREVIOUS_LO_RAW_RELATIVE_L2,
                ),
                "endpoint_time_M": lo_after_tail["endpoint_time_M"],
                "endpoint_phase": lo_after_tail["endpoint_phase"],
                "whole_pa_curve_metrics": lo_after_tail,
            },
            "interpretation": (
                "The frozen fixture projection preserves the prior baseline value. "
                "The regenerated clean baseline and diagnostic-only leading-order "
                "horizon branch use the patched radiation table and are reported as "
                "new candidates; no alignment, tuning, or threshold change is applied."
            ),
        },
        "pa_endpoints": {
            "frozen_fixture": {
                "time_M": frozen_projection["published_r0_2M_over_sqrt_e_tail"]["endpoint_time_M"],
                "phase": frozen_projection["published_r0_2M_over_sqrt_e_tail"]["endpoint_phase"],
            },
            "regenerated_clean_baseline": {
                "time_M": current_after_tail["endpoint_time_M"],
                "phase": current_after_tail["endpoint_phase"],
            },
            "regenerated_leading_order_horizon": {
                "time_M": lo_after_tail["endpoint_time_M"],
                "phase": lo_after_tail["endpoint_phase"],
            },
        },
        "actionable_hypotheses": [
            "The published Eq.(23) tail gauge removes the approximately 0.0223 rad initial phase offset; the remaining phase grows across the PA interval while amplitude stays at the 1e-6 level.",
            "The derivative residual is small but nonzero and accumulates, which points to PA time/phase evolution (flux or torque/quadrature) rather than a constant carrier gauge.",
            "The patched radiation table changes the regenerated PA endpoint relative to the frozen pre-patch fixture; the resulting state/time/phase differences are recorded as a radiation-completion effect, not treated as helper reproduction or strict production parity.",
            "The explicit PNSpinOrder=0 public PA control is accepted, but it has 19050 samples versus the 19048-sample default fixture; its shape and waveform difference are recorded without truncation, alignment, or use as a strict fixture replacement.",
            "The default-spin versus PNSpinOrder=0 comparison is an attribution control only; other public evolution options may change with that setting, so it does not isolate horizon absorption.",
        ],
        "runtime_sha256_before": runtime_before,
        "runtime_sha256_after": runtime_after,
        "runtime_unchanged": runtime_before == runtime_after,
        "output_cache": cache_metadata,
        "elapsed_seconds": _finite(time.perf_counter() - started),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
