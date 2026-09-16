#!/usr/bin/env python3
"""Reassemble cached TEOB trajectories with native NQC and ringdown code.

The convergence cache stores accepted ``(time, state)`` knots, but not the
derived frequency or NQC work arrays. This diagnostic evaluates the same
clean TEOB vector field at those cached states, then runs the native waveform,
NQC, and ringdown assemblers. The public waveform is used only for scoring;
it never supplies output samples or correction factors.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import os
import platform
import shlex
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_teob_paired_driver_clean as driver
from torchwave.approximants.eob.teob_vacuum_driver_clean import TEOBVacuumRadiationDriver
from torchwave.approximants.eob import teob_ringdown_assembly_clean as ringdown
from torchwave.approximants.eob import teob_merger_nqc_clean as nqc_module

DEFAULT_CACHE = ROOT / "docs/reviews/2026-09-15-no-contact/teob-ode-convergence-clean/convergence-arrays.npz"
DEFAULT_REPORT = ROOT / "docs/reviews/2026-09-15-no-contact/teob-ode-convergence-clean/report.json"
DEFAULT_OUTPUT = ROOT / "docs/reviews/2026-09-15-parity-strategy-experiment/eob/native-fixed-event-reassembly.json"
LEVELS = ("baseline", "tight_0p1", "tight_0p01")


_FIXED_NQC = None


def _fixed_nqc_function():
    """Build a diagnostic-only clone with exact grid/event override hooks."""
    global _FIXED_NQC
    if _FIXED_NQC is not None:
        return _FIXED_NQC
    source = inspect.getsource(nqc_module.assemble_teob_merger_nqc)
    old_sig = "    max_condition: float = 1.0e10,\n) -> TEOBMergerNQCResult:"
    new_sig = "    max_condition: float = 1.0e10,\n    auxiliary_time_override: Optional[Tensor] = None,\n    fixed_peak_time: Optional[Tensor] = None,\n    fixed_nqc_time: Optional[Tensor] = None,\n) -> TEOBMergerNQCResult:"
    assert old_sig in source
    source = source.replace(old_sig, new_sig, 1)
    old_grid = "    node_count_tensor = torch.floor(support_width / spacing) + 1.0\n    if not bool(torch.isfinite(node_count_tensor)):\n        raise TEOBMergerNQCError(\"auxiliary node count is not finite\")\n    node_count = int(node_count_tensor.detach().item())\n    if node_count < 5:\n        raise TEOBMergerNQCError(\"auxiliary grid has insufficient five-knot derivative support\")\n    aux_time = support_time[0] + torch.arange(node_count, dtype=dtype, device=device) * spacing"
    new_grid = "    if auxiliary_time_override is None:\n        node_count_tensor = torch.floor(support_width / spacing) + 1.0\n        if not bool(torch.isfinite(node_count_tensor)):\n            raise TEOBMergerNQCError(\"auxiliary node count is not finite\")\n        node_count = int(node_count_tensor.detach().item())\n        if node_count < 5:\n            raise TEOBMergerNQCError(\"auxiliary grid has insufficient five-knot derivative support\")\n        aux_time = support_time[0] + torch.arange(node_count, dtype=dtype, device=device) * spacing\n    else:\n        aux_time = _as_real(auxiliary_time_override, name=\"auxiliary_time_override\", dtype=dtype, device=device)\n        if aux_time.ndim != 1 or aux_time.numel() < 5 or not bool(torch.isfinite(aux_time).all()):\n            raise TEOBMergerNQCError(\"auxiliary_time_override must be a finite one-dimensional grid\")\n        if bool((aux_time[1:] <= aux_time[:-1]).any()):\n            raise TEOBMergerNQCError(\"auxiliary_time_override must be strictly increasing\")\n        if bool((aux_time < support_time[0]).any()) or bool((aux_time > support_time[-1]).any()):\n            raise TEOBMergerNQCError(\"auxiliary_time_override lies outside spline support\")\n        spacing = aux_time[1] - aux_time[0]"
    assert old_grid in source
    source = source.replace(old_grid, new_grid, 1)
    old_event = "    t_peak_grid = aux_time[peak_index]\n    t_nqc = t_peak_grid - torch.as_tensor(1.0, dtype=dtype, device=device)"
    new_event = "    t_peak_grid = aux_time[peak_index] if fixed_peak_time is None else _as_real(fixed_peak_time, name=\"fixed_peak_time\", dtype=dtype, device=device, scalar=True)\n    t_nqc = (t_peak_grid - torch.as_tensor(1.0, dtype=dtype, device=device)\n             if fixed_nqc_time is None else _as_real(fixed_nqc_time, name=\"fixed_nqc_time\", dtype=dtype, device=device, scalar=True))"
    assert old_event in source
    source = source.replace(old_event, new_event, 1)
    namespace = vars(nqc_module).copy()
    exec(compile(source, "<native_fixed_nqc_clone>", "exec"), namespace)
    _FIXED_NQC = namespace["assemble_teob_merger_nqc"]
    return _FIXED_NQC


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arr(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError(f"metric shape mismatch: {candidate.shape} != {reference.shape}")
    d = candidate - reference
    return {"samples": int(candidate.size),
            "relative_l2": float(np.linalg.norm(d) / max(np.linalg.norm(reference), 1e-300)),
            "max_abs": float(np.max(np.abs(d))),
            "rms_abs": float(np.sqrt(np.mean(np.abs(d) ** 2)))}


def _projection_scale() -> float:
    try:
        import lal  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("lal constants are required for the reviewed projection") from exc
    return -math.sqrt(5.0 / (4.0 * math.pi)) * (
        50.0 * lal.MTSUN_SI * lal.C_SI / (driver.DISTANCE_MPC * 1e6 * lal.PC_SI)
    )


def _native_frequency(hamiltonian: Any, radiation: Any, state: torch.Tensor) -> torch.Tensor:
    """Recover Omega exactly as continuation does: vector_field[..., 1]."""
    return TEOBVacuumRadiationDriver(hamiltonian, radiation).vector_field(state)[..., 1]


@contextlib.contextmanager
def _forced_peak(peak_time: torch.Tensor | None):
    """Locally override only ringdown peak time; retain native assembly."""
    if peak_time is None:
        yield
        return
    original = ringdown._peak_polynomial

    def wrapped(time, frequency, index, spacing):
        _, first, second, coefficients = original(time, frequency, index, spacing)
        return peak_time.to(dtype=time.dtype, device=time.device), first, second, coefficients

    ringdown._peak_polynomial = wrapped
    try:
        yield
    finally:
        ringdown._peak_polynomial = original


def _assemble(nqc: Any, time: torch.Tensor,
              nu: torch.Tensor, final_mass: torch.Tensor, final_spin: torch.Tensor,
              event_peak_override: float | None = None) -> Any:
    matching_state = nqc.auxiliary.state[nqc.matching_index]
    peak_override = None
    if event_peak_override is not None:
        peak_override = torch.as_tensor(event_peak_override, dtype=time.dtype)
    with _forced_peak(peak_override):
        return driver.assemble_teob_ringdown_22(
            time, nqc.original.corrected_amplitude, nqc.original.corrected_phase,
            nqc.auxiliary.time, nqc.auxiliary.corrected_amplitude,
            nqc.auxiliary.corrected_phase, nqc.auxiliary.orbital_frequency,
            nu=nu, final_mass=final_mass, final_spin=final_spin,
            matching_time=nqc.t_nqc, matching_state=matching_state,
            output_step=torch.as_tensor(driver.OUTPUT_STEP, dtype=time.dtype),
        )


def _score(result: Any, public_time: np.ndarray, public_mode: np.ndarray,
           projection: float, final_mass: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
    # Public mode is read only by metrics below; output is built from native
    # signed amplitude and lifted phase on the native extended support.
    xt, amp, phase = arr(result.extended_time), arr(result.extended_amplitude), arr(result.extended_phase)
    mask = public_time <= xt[-1]
    score_time, score_reference = public_time[mask], public_mode[mask]
    mode = projection * driver._natural_cubic_numpy(xt, amp, score_time) * np.exp(
        -1j * driver._natural_cubic_numpy(xt, phase, score_time)
    )
    event = driver._events(result, public_time, final_mass)
    event.update({"metrics_vs_public": metrics(mode, score_reference),
                  "scored_public_samples": int(score_time.size),
                  "full_public_support": bool(mask.all()),
                  "signed_extended_negative_samples": int(np.count_nonzero(amp < 0.0))})
    return mode, event


def _common_auxiliary_grid(data: dict[str, np.ndarray], levels: tuple[str, ...],
                           spacing: float = 0.1) -> torch.Tensor:
    """Choose one physical grid inside every level's native spline support."""
    starts, ends = [], []
    for level in levels:
        time = data[f"{level}_trajectory_time"]
        cut = int(np.searchsorted(time, time[-1] - 20.0, side="right")) - 1
        cut = max(0, min(cut, time.size - 2))
        starts.append(float(time[cut]))
        ends.append(float(time[-2]))
    start, end = max(starts), min(ends)
    count = int(math.floor((end - start) / spacing)) + 1
    if count < 5:
        raise ValueError("common auxiliary support has fewer than five knots")
    return torch.as_tensor(start + spacing * np.arange(count), dtype=torch.float64)


def _one(level: str, data: dict[str, np.ndarray], public_time: np.ndarray,
         public_mode: np.ndarray, projection: float, spacing: float,
         event_shift: float | None = None, perturb: float = 0.0,
         event_peak_override: float | None = None,
         auxiliary_time_override: torch.Tensor | None = None,
         fixed_nqc_peak: float | None = None,
         fixed_nqc_time: float | None = None,
         use_fixed_clone: bool = False) -> dict[str, Any]:
    dtype = torch.float64
    time_t = torch.as_tensor(data[f"{level}_trajectory_time"], dtype=dtype)
    state_np = data[f"{level}_trajectory_state"].copy()
    if perturb:
        shape = np.linspace(-1.0, 1.0, state_np.shape[0])
        state_np[:, 0] *= 1.0 + perturb * shape
    state = torch.as_tensor(state_np, dtype=dtype)
    hamiltonian = driver._hamiltonian()
    radiation = driver._radiation_reaction(hamiltonian, "current")
    with torch.no_grad():
        frequency = _native_frequency(hamiltonian, radiation, state)
        mode = driver.teob_waveform_state_mode(
            state, hamiltonian, 2, 2, Omega=frequency, residual_phase="zero-unpublished")
        amplitude, phase = mode.abs(), driver.phase_from_mode(mode, state[:, 1])
        radial_acceleration = driver.conservative_trajectory_quantities(
            hamiltonian, state).radial_acceleration
        nu = torch.tensor(driver.NU_VALUE, dtype=dtype)
        targets = driver.published_2017_nqc_targets(nu, torch.tensor(0.0, dtype=dtype))
        nqc_builder = _fixed_nqc_function() if (
            use_fixed_clone or auxiliary_time_override is not None
            or fixed_nqc_peak is not None or fixed_nqc_time is not None
        ) else driver.assemble_teob_merger_nqc
        nqc_kwargs = {}
        if auxiliary_time_override is not None:
            nqc_kwargs["auxiliary_time_override"] = auxiliary_time_override
        if fixed_nqc_peak is not None:
            nqc_kwargs["fixed_peak_time"] = torch.as_tensor(fixed_nqc_peak, dtype=dtype)
        if fixed_nqc_time is not None:
            nqc_kwargs["fixed_nqc_time"] = torch.as_tensor(fixed_nqc_time, dtype=dtype)
        nqc = nqc_builder(
            time_t, state, amplitude=amplitude, phase=phase,
            orbital_frequency=frequency, radial_acceleration=radial_acceleration,
            nu=nu, fixed_step=torch.tensor(spacing, dtype=dtype), targets=targets,
            target_provenance=driver.PUBLISHED_2017_NQC_PROVENANCE, **nqc_kwargs)
        final_mass, final_spin = driver.uib2016_nonspinning_remnant(
            driver.MASS1_MSUN, driver.MASS2_MSUN)
        final_mass, final_spin = final_mass.to(dtype=dtype), final_spin.to(dtype=dtype)
        if event_shift is not None and event_peak_override is None:
            native_result = _assemble(nqc, time_t, nu, final_mass, final_spin)
            event_peak_override = float(native_result.events.peak_time) + event_shift
        result = _assemble(nqc, time_t, nu, final_mass, final_spin, event_peak_override)
    output, event = _score(result, public_time, public_mode, projection, final_mass)
    return {"level": level, "spacing_M": spacing, "event_shift_M": event_shift,
            "state_perturbation": perturb, "trajectory_samples": int(time_t.numel()),
            "frequency_reconstructed_from_vector_field": True,
            "auxiliary_time": arr(nqc.auxiliary.time).tolist(),
            "nqc": {"matching_index_auxiliary": int(nqc.matching_index),
                    "matching_time_M": float(nqc.t_nqc), "peak_grid_time_M": float(nqc.t_peak_grid),
                    "auxiliary_samples": int(nqc.auxiliary.time.numel()),
                    "auxiliary_spacing_M": float(nqc.auxiliary_spacing),
                    "amplitude_condition": float(nqc.match.amplitude_condition),
                    "phase_condition": float(nqc.match.phase_condition),
                    "amplitude_coefficients": arr(nqc.coefficients.amplitude).tolist(),
                    "phase_coefficients": arr(nqc.coefficients.phase).tolist(),
                    "amplitude_matrix": arr(nqc.match.amplitude_matrix).tolist(),
                    "phase_matrix": arr(nqc.match.phase_matrix).tolist(),
                    "amplitude_residual": arr(nqc.match.amplitude_residual).tolist(),
                    "phase_residual": arr(nqc.match.phase_residual).tolist(),
                    "negative_corrected_amplitude_samples": int(nqc.auxiliary_negative_amplitude.sum())},
            "events": event, "output": output,
            "native_output_metrics_vs_public": event["metrics_vs_public"]}


def run(cache_path: Path = DEFAULT_CACHE, report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    started = time.perf_counter()
    # Read the prior report only to bind the cache provenance; no saved mode is
    # used to construct a replay.
    json.loads(report_path.read_text())
    with np.load(cache_path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    public_time, public_mode = data["public_time"], data["public_mode"]
    projection = _projection_scale()
    baseline = _one("baseline", data, public_time, public_mode, projection, 0.1)
    baseline_reproduction = metrics(baseline["output"], data["baseline_processed_mode"])
    fixed_event_peak = baseline["events"]["peak_time_initial_mass"]
    controls = []
    for level in LEVELS:
        for spacing in (0.05, 0.1, 0.2):
            row = _one(level, data, public_time, public_mode, projection, spacing,
                       event_peak_override=fixed_event_peak)
            row.pop("output", None)
            row["controls_held_fixed"] = {"event_peak_time_M": fixed_event_peak}
            controls.append(row)
    event_controls = []
    for shift in (-0.05, 0.0, 0.05):
        row = _one("baseline", data, public_time, public_mode, projection, 0.1,
                   event_shift=shift)
        row.pop("output", None)
        event_controls.append(row)
    pair_levels = ("tight_0p1", "tight_0p01")
    common_grid = _common_auxiliary_grid(data, pair_levels, spacing=0.1)
    native_pair = {
        level: _one(level, data, public_time, public_mode, projection, 0.1)
        for level in pair_levels
    }
    fixed_nqc_peak = native_pair["tight_0p1"]["nqc"]["peak_grid_time_M"]
    fixed_nqc_time = native_pair["tight_0p1"]["nqc"]["matching_time_M"]
    fixed_rd_peak = native_pair["tight_0p1"]["events"]["peak_time_initial_mass"]
    pair_variants: dict[str, dict[str, Any]] = {}
    for variant in ("native", "common_grid", "fixed_events", "common_grid_fixed_events"):
        rows = {}
        for level in pair_levels:
            kwargs: dict[str, Any] = {}
            if variant in ("common_grid", "common_grid_fixed_events"):
                kwargs["auxiliary_time_override"] = common_grid
            if variant in ("fixed_events", "common_grid_fixed_events"):
                kwargs.update({"fixed_nqc_peak": fixed_nqc_peak,
                               "fixed_nqc_time": fixed_nqc_time,
                               "event_peak_override": fixed_rd_peak})
            rows[level] = _one(level, data, public_time, public_mode, projection, 0.1, **kwargs)
        pair_variants[variant] = rows
    clone_levels = ("baseline", *pair_levels)
    native_reference = {"baseline": baseline, **native_pair}
    clone_native = {
        level: _one(level, data, public_time, public_mode, projection, 0.1,
                    use_fixed_clone=True)
        for level in clone_levels
    }
    pairwise = {}
    for variant, rows in pair_variants.items():
        left, right = rows[pair_levels[0]], rows[pair_levels[1]]
        if left["output"].shape != right["output"].shape:
            raise RuntimeError(f"controlled {variant} outputs have different support")
        pairwise[variant] = metrics(left["output"], right["output"])
    clone_checks = {
        level: metrics(clone_native[level]["output"], native_reference[level]["output"])
        for level in clone_levels
    }
    source_files = {
        "replay_script": Path(__file__).resolve(),
        "native_nqc_module": Path(nqc_module.__file__).resolve(),
        "native_ringdown_module": Path(ringdown.__file__).resolve(),
        "clean_driver": Path(driver.__file__).resolve(),
    }
    common_grid_equal = np.array_equal(
        np.asarray(pair_variants["common_grid"][pair_levels[0]]["auxiliary_time"]),
        np.asarray(pair_variants["common_grid"][pair_levels[1]]["auxiliary_time"]),
    )
    fixed_event_equal = all(
        pair_variants["common_grid_fixed_events"][level]["nqc"]["peak_grid_time_M"] == fixed_nqc_peak
        and pair_variants["common_grid_fixed_events"][level]["nqc"]["matching_time_M"] == fixed_nqc_time
        and pair_variants["common_grid_fixed_events"][level]["events"]["peak_time_initial_mass"] == fixed_rd_peak
        for level in pair_levels
    )
    base = _one("baseline", data, public_time, public_mode, projection, 0.1)
    perturb_rows = []
    for eps in (-1e-8, 1e-8):
        row = _one("baseline", data, public_time, public_mode, projection, 0.1, perturb=eps)
        d = metrics(row["output"], base["output"])
        perturb_rows.append({"epsilon": eps, "native_output_delta_from_same_baseline": d,
                             "amplification": d["relative_l2"] / abs(eps),
                             "branch_crossed": row["nqc"]["negative_corrected_amplitude_samples"] != base["nqc"]["negative_corrected_amplitude_samples"]})
    return {"status": "complete", "experiment": "native cached-trajectory fixed-event reassembly",
            "run": {"host": socket.gethostname(), "platform": platform.platform(), "cwd": str(Path.cwd()),
                    "command": shlex.join([sys.executable, *sys.argv]), "pid": os.getpid(),
                    "elapsed_seconds": time.perf_counter() - started},
            "inputs": {
                "cache": sha256(cache_path),
                "prior_report": sha256(report_path),
                "source_files": {
                    name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in source_files.items()
                },
            },
            "baseline_reproduction": {"native_vs_saved_processed": baseline_reproduction,
                                      "native_vs_public": baseline["native_output_metrics_vs_public"]},
            "controls": controls, "event_controls": event_controls,
            "controlled_pair": {
                "levels": list(pair_levels),
                "common_auxiliary_grid": common_grid.tolist(),
                "common_grid_equal": common_grid_equal,
                "fixed_event_times_equal": fixed_event_equal,
                "fixed_nqc_peak_time_M": fixed_nqc_peak,
                "fixed_nqc_time_M": fixed_nqc_time,
                "fixed_rd_peak_time_M": fixed_rd_peak,
                "pairwise_mode_metrics": pairwise,
                "native_clone_vs_native": clone_checks,
                "variants": {
                    name: {level: {k: v for k, v in row.items() if k != "output"}
                           for level, row in rows.items()}
                    for name, rows in pair_variants.items()
                },
            },
            "perturbation": {"rows": perturb_rows},
            "limits": [
                "Frequency is reconstructed by evaluating the native clean vector field at cached states; no ODE reintegration occurs.",
                "The event probe locally overrides only the native peak polynomial time, then runs native ringdown assembly.",
                "Public mode is used only as a scoring reference. No oracle coefficient fitting or source reads were used.",
                "The saved processed cache contains public-grid output; reproduction compares against it after the same native interpolation.",
            ]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)
    result = run(args.cache, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "controls": len(result["controls"])}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
