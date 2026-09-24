"""Reproduce pinned-source v4 flux and initial-condition component checks.

This development diagnostic never supplies LAL code to the native generator.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import lal
import lalsimulation
import torch

from . import qualify_generator as q
from .diagnose_hamiltonian_same_state import checked_file, digest, git
from .diagnose_lal_dynamics import _reference as public_dynamics
from .eob.torch.source_calibration_hamiltonian import SourceCalibratedHistoricalHamiltonian
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_mode22_from_hamiltonian,
)


ROOT = q.ROOT
PLAN = ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json"
MODES = tuple((ell, m) for ell in range(2, 9) for m in range(1, ell+1))


def _snapshot(repo: Path, spec: dict, private: Path, headers: Path) -> None:
    tag = spec["tag_object"]
    if (git(repo, "rev-parse", spec["tag"]).decode().strip() != tag
            or git(repo, "cat-file", "-t", tag).decode().strip() != "tag"
            or git(repo, "rev-parse", f"{tag}^{{commit}}").decode().strip()
            != spec["peeled_commit"]):
        raise ValueError("Pinned LAL source tag changed")
    for relative, expected in spec["files_sha256"].items():
        content = git(repo, "show", f"{tag}:{relative}")
        if digest(content) != expected:
            raise ValueError(f"Pinned source changed: {relative}")
        target = private / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for relative, expected in spec["public_headers_sha256"].items():
        content = git(repo, "show", f"{tag}:{relative}")
        if digest(content) != expected:
            raise ValueError(f"Pinned public header changed: {relative}")
        checked_file(headers / "lal" / Path(relative).name, expected)
    for relative, expected in spec["generated_headers_sha256"].items():
        checked_file(headers / relative, expected)


def _libraries(spec: dict) -> dict[str, Path]:
    if (importlib.metadata.version("lalsuite") != spec["lalsuite_distribution"]
            or lal.__version__ != spec["lal_version"]
            or lalsimulation.__version__ != spec["lalsimulation_version"]
            or lalsimulation.SimulationVCSInfo.vcsId != spec["vcs_id"]
            or lalsimulation.SimulationVCSInfo.vcsTag != spec["vcs_tag"]
            or lalsimulation.SimulationVCSInfo.vcsClean != "CLEAN"):
        raise ValueError("Pinned wheel version changed")
    library_dir = Path(lal.__file__).resolve().parent / ".dylibs"
    extension = list(Path(lalsimulation.__file__).resolve().parent.glob(
        "_lalsimulation*.so"))
    if len(extension) != 1:
        raise ValueError("Expected exactly one lalsimulation Python extension")
    checked_file(extension[0], spec["python_extension_sha256"])
    libraries = {name: library_dir / name for name in spec["libraries_sha256"]}
    for name, path in libraries.items():
        checked_file(path, spec["libraries_sha256"][name])
    return libraries


def _compile(private: Path, headers: Path, gsl: Path,
             libraries: dict[str, Path], adapter: Path, output: Path,
             source_includes: list[str], public_includes: list[str],
             optimization: str = "-O2", require_wheel_flux: bool = True,
             extra_flags: tuple[str, ...] = ()) -> None:
    if (optimization not in ("-O0", "-O2", "-O3")
            or extra_flags not in ((), ("-ffast-math",))):
        raise ValueError("Unsupported compiler comparison")
    command = [
        "clang", "-dynamiclib", optimization, *extra_flags, "-H",
        "-Wno-gnu-folding-constant",
        "-Wl,-headerpad_max_install_names",
        "-I", str(private / "lalsimulation/lib"),
        "-I", str(private / "lal/lib/factories"),
        "-I", str(headers), "-I", str(gsl),
        "-o", str(output), str(adapter),
        *(str(path) for path in libraries.values()),
    ]
    build = subprocess.run(command, check=True, capture_output=True, text=True)
    for relative in source_includes:
        if str(private / relative) not in build.stderr:
            raise ValueError(f"Compiler did not include pinned {relative}")
    for relative in public_includes:
        if str(headers / relative) not in build.stderr:
            raise ValueError(f"Compiler did not include checked header {relative}")
    if "lalsuite-git" in build.stderr:
        raise ValueError("Compiler read from mutable LAL checkout")
    changes = []
    for name, path in libraries.items():
        changes.extend(("-change", f"/DLC/lal/.dylibs/{name}", str(path)))
    subprocess.run(["install_name_tool", *changes, str(output)], check=True,
                   capture_output=True, text=True)
    linked = subprocess.run(["otool", "-L", str(output)], check=True,
                            capture_output=True, text=True).stdout
    for path in libraries.values():
        if str(path) not in linked:
            raise ValueError(f"Adapter did not link pinned wheel library {path}")
    bindings = subprocess.run(["nm", "-m", str(output)], check=True,
                              capture_output=True, text=True).stdout
    if (require_wheel_flux and "_XLALSimInspiralEOBPAFluxWrapper "
            "(from liblalsimulation.37)" not in bindings):
        raise ValueError("PA flux wrapper was not imported from pinned wheel")
    if (not require_wheel_flux and "_XLALSimIMRSpinAlignedEOBModes "
            "(from liblalsimulation.37)" in bindings):
        raise ValueError("Full source generator resolved to the wheel")


def _native_flux(row: dict, radius: float, pphi: float,
                 omega: float, choices: dict) -> float:
    def scalar(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64)
    x1 = row["mass1"] / (row["mass1"]+row["mass2"])
    x2 = 1.-x1
    ham = SourceCalibratedHistoricalHamiltonian(
        scalar(x1*x2), scalar(row["chi1"]), scalar(row["chi2"]),
        delta=scalar(x1-x2))
    flux = SEOBNRv4PublishedFlux(**choices)
    state = torch.stack((scalar(radius), scalar(0.), scalar(0.), scalar(pphi)))
    with torch.no_grad():
        inputs = flux.from_hamiltonian(state, ham, omega=scalar(omega))
        return float(flux.energy_flux(state, **inputs)/(x1*x2))


def _native_mode_and_flux(row: dict, radius: float, phi: float, pr_star: float,
                          pphi: float, omega: float,
                          flux_choices: dict,
                          mode_choices: dict) -> tuple[complex, float, float, list[float]]:
    def scalar(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64)
    x1 = row["mass1"]/(row["mass1"]+row["mass2"])
    x2 = 1.-x1
    ham = SourceCalibratedHistoricalHamiltonian(
        scalar(x1*x2), scalar(row["chi1"]), scalar(row["chi2"]),
        delta=scalar(x1-x2))
    flux = SEOBNRv4PublishedFlux(**flux_choices)
    state = torch.stack((scalar(radius), scalar(phi), scalar(pr_star), scalar(pphi)))
    with torch.no_grad():
        mode = seobnrv4_published_mode22_from_hamiltonian(
            state, ham, flux, omega=scalar(omega), **mode_choices)
        inputs = flux.from_hamiltonian(state, ham, omega=scalar(omega))
        modes = flux._signed_amplitudes(state, **inputs)
        components = [float((m*omega)**2*modes[ell, m].square()
                            /(8*math.pi*x1*x2)) for ell, m in MODES]
        total = float(flux.energy_flux(state, **inputs)/(x1*x2))
        if abs(sum(components)/total-1.) > 1e-12:
            raise ValueError("Native flux modes do not sum to total")
        return complex(mode.item()), float(inputs["hreal_over_M"]), total, components


def _mode_flux_comparison(source_values, native_components: list[float],
                          sum_tolerance: float) -> dict:
    source_total = source_values[3]
    source_components = source_values[4:]
    if (len(source_components) != len(MODES)
            or abs(sum(source_components)/source_total-1.) > sum_tolerance):
        raise ValueError("Pinned-source flux modes do not sum to total")
    rows = []
    for (ell, m), source, native in zip(MODES, source_components, native_components):
        rows.append({"ell": ell, "m": m, "source_over_nu": source,
                     "native_over_nu": native,
                     "native_minus_source_over_nu": native-source,
                     "relative_to_source_total": (native-source)/source_total})
    rows.sort(key=lambda item: abs(item["native_minus_source_over_nu"]), reverse=True)
    return {"source_sum_relative_to_total": sum(source_components)/source_total-1.,
            "components_by_absolute_residual": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise RuntimeError("Pinned wheel C-link recipe is macOS-only")
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-flux-initial-plan.v6"
            or plan["release_qualified"] is not False):
        raise ValueError("Unexpected diagnostic plan")
    if (plan["pinned_source"]["peeled_commit"]
            != plan["pinned_wheel"]["vcs_id"]
            or plan["pinned_source"]["tag"]
            != plan["pinned_wheel"]["vcs_tag"]):
        raise ValueError("Source snapshot does not match the installed wheel")
    adapter_spec = plan["adapter"]
    adapter = ROOT / adapter_spec["path"]
    checked_file(adapter, adapter_spec["sha256"])
    full_adapter_spec = plan["full_generator_adapter"]
    full_adapter = ROOT / full_adapter_spec["path"]
    checked_file(full_adapter, full_adapter_spec["sha256"])
    for relative, expected in plan["native_files_sha256"].items():
        checked_file(ROOT / relative, expected)
    public_spec = plan["public_initial_evidence"]
    public_path = ROOT / public_spec["path"]
    checked_file(public_path, public_spec["sha256"])
    source_spec = plan["source_calibration_initial_evidence"]
    source_path = ROOT / source_spec["path"]
    checked_file(source_path, source_spec["sha256"])
    public = json.loads(public_path.read_text())
    calibration = json.loads(source_path.read_text())
    source_choice = dict(calibration["flux_choice"],
                         **plan["source_placement_flux_choice_overrides"])
    trajectory_spec = plan["mode_trajectory_evidence"]
    trajectory_path = ROOT / trajectory_spec["path"]
    checked_file(trajectory_path, trajectory_spec["sha256"])
    trajectory = json.loads(trajectory_path.read_text())
    if (public["status"] != "diagnostic_complete"
            or public_spec["row_indices"] != [0, 1]
            or len(calibration["rows"]) != 2):
        raise ValueError("Frozen initial-state rows changed")
    libraries = _libraries(plan["pinned_wheel"])
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="diffgw-v4-flux-initial-") as tmp:
        private = Path(tmp) / "source"
        _snapshot(args.source_root, plan["pinned_source"], private, args.headers)
        output = Path(tmp) / "adapter.dylib"
        _compile(private, args.headers, args.gsl_include, libraries, adapter,
                 output, plan["flux_source_includes"],
                 plan["flux_public_includes"])
        library = ctypes.CDLL(str(output))
        function = library.diffgw_v4_flux_same_state
        function.argtypes = [ctypes.c_double]*7 + [ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        initial_function = library.diffgw_v4_initial_conditions
        initial_function.argtypes = [ctypes.c_double]*5 + [ctypes.c_int,
                                                           ctypes.POINTER(ctypes.c_double)]
        initial_function.restype = ctypes.c_int
        mode_function = library.diffgw_v4_mode22_same_state
        mode_function.argtypes = [ctypes.c_double]*9 + [
            ctypes.POINTER(ctypes.c_double)]
        mode_function.restype = ctypes.c_int
        unoptimized_output = Path(tmp) / "adapter_O0.dylib"
        _compile(private, args.headers, args.gsl_include, libraries, adapter,
                 unoptimized_output, plan["flux_source_includes"],
                 plan["flux_public_includes"], "-O0")
        unoptimized_library = ctypes.CDLL(str(unoptimized_output))
        unoptimized_initial = unoptimized_library.diffgw_v4_initial_conditions
        unoptimized_initial.argtypes = initial_function.argtypes
        unoptimized_initial.restype = ctypes.c_int
        fast_output = Path(tmp) / "adapter_fast_math.dylib"
        _compile(private, args.headers, args.gsl_include, libraries, adapter,
                 fast_output, plan["flux_source_includes"],
                 plan["flux_public_includes"], extra_flags=("-ffast-math",))
        fast_library = ctypes.CDLL(str(fast_output))
        fast_initial = fast_library.diffgw_v4_initial_conditions
        fast_initial.argtypes = initial_function.argtypes
        fast_initial.restype = ctypes.c_int
        full_output = Path(tmp) / "full_generator.dylib"
        _compile(private, args.headers, args.gsl_include, libraries,
                 full_adapter, full_output,
                 plan["full_generator_source_includes"],
                 plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        full_library = ctypes.CDLL(str(full_output))
        full_first = full_library.diffgw_v4_full_source_first
        full_first.argtypes = initial_function.argtypes[:-2] + [
            ctypes.POINTER(ctypes.c_double)]
        full_first.restype = ctypes.c_int
        rows = []
        for index in public_spec["row_indices"]:
            frozen = calibration["rows"][index]
            row = frozen["row"]
            state = frozen["choices"]["lal_v4_rounded"]
            mass = row["mass1"]+row["mass2"]
            out = (ctypes.c_double*9)()
            rc = function(row["mass1"]/mass, row["mass2"]/mass,
                          row["chi1"], row["chi2"], state["r"], state["pphi"],
                          frozen["native_start_Momega"], out)
            if rc:
                raise RuntimeError(f"Pinned source adapter failed at row {index}: {rc}")
            initial = {}
            for name, optimized in (("legacy", 0), ("optimized", 1)):
                values = (ctypes.c_double*3)()
                rc = initial_function(row["mass1"], row["mass2"],
                                      row["chi1"], row["chi2"],
                                      frozen["native_start_Momega"], optimized, values)
                if rc:
                    raise RuntimeError(f"Pinned source {name} IC failed at row {index}: {rc}")
                initial[name] = dict(zip(("r", "pr_star", "pphi"), values))
            unoptimized = {}
            fast_math = {}
            for name, optimized in (("legacy", 0), ("optimized", 1)):
                values = (ctypes.c_double*3)()
                rc = unoptimized_initial(row["mass1"], row["mass2"],
                                         row["chi1"], row["chi2"],
                                         frozen["native_start_Momega"], optimized, values)
                if rc:
                    raise RuntimeError(
                        f"Pinned source {name} O0 IC failed at row {index}: {rc}")
                unoptimized[name] = dict(zip(("r", "pr_star", "pphi"), values))
                fast_values = (ctypes.c_double*3)()
                rc = fast_initial(row["mass1"], row["mass2"],
                                  row["chi1"], row["chi2"],
                                  frozen["native_start_Momega"], optimized,
                                  fast_values)
                if rc:
                    raise RuntimeError(
                        f"Pinned source {name} fast-math IC failed at row {index}: {rc}")
                fast_math[name] = dict(zip(("r", "pr_star", "pphi"),
                                          fast_values))
            full_values = (ctypes.c_double*3)()
            rc = full_first(row["mass1"], row["mass2"], row["chi1"],
                            row["chi2"], row["omega_start"], full_values)
            if rc:
                raise RuntimeError(
                    f"Pinned full generator failed at row {index}: {rc}")
            full_initial = dict(zip(("r", "pr_star", "pphi"), full_values))
            if full_initial != initial["legacy"]:
                raise ValueError(
                    f"Source generator/direct IC mismatch at row {index}")
            source_flux, wheel_flux = out[0], out[1]
            if abs(wheel_flux/source_flux-1.) > plan["source_wheel_max_relative_flux_difference"]:
                raise ValueError(f"Source/wheel flux mismatch at row {index}")
            native_flux = _native_flux(row, state["r"], state["pphi"],
                                       frozen["native_start_Momega"],
                                       calibration["flux_choice"])
            mode_values = (ctypes.c_double*39)()
            rc = mode_function(
                row["mass1"]/mass, row["mass2"]/mass,
                row["chi1"], row["chi2"], state["r"], 0.,
                0., state["pphi"], frozen["native_start_Momega"],
                mode_values)
            if rc:
                raise RuntimeError(
                    f"Pinned source same-state mode failed at row {index}: {rc}")
            source_mode = complex(mode_values[0], mode_values[1])
            native_mode, native_hreal, native_mode_flux, native_components = _native_mode_and_flux(
                row, state["r"], 0., 0., state["pphi"],
                frozen["native_start_Momega"], calibration["flux_choice"],
                plan["native_mode_choices"])
            if (abs(native_mode-source_mode)/abs(source_mode)
                    > plan["native_source_mode_max_relative_difference"]
                    or abs(native_hreal-mode_values[2])
                    > plan["native_source_Hreal_max_absolute_difference"]):
                raise ValueError(f"Native/source raw mode mismatch at row {index}")
            mode_flux = _mode_flux_comparison(
                mode_values, native_components,
                plan["source_flux_mode_sum_max_relative_difference"])
            _, _, source_placement_flux, source_placement_components = (
                _native_mode_and_flux(
                    row, state["r"], 0., 0., state["pphi"],
                    frozen["native_start_Momega"], source_choice,
                    plan["native_mode_choices"]))
            if (abs(source_placement_flux/mode_values[3]-1.)
                    > plan["native_source_placement_flux_max_relative_difference"]):
                raise ValueError(f"Source-placement flux mismatch at row {index}")
            source_placement_mode_flux = _mode_flux_comparison(
                mode_values, source_placement_components,
                plan["source_flux_mode_sum_max_relative_difference"])
            observed = frozen["public_initial"]
            public_first_states = {}
            for step in plan["public_first_state_steps_M"]:
                first = public_dynamics(row, step)[1:, 0]
                state_by_grid = dict(zip(("r", "phi", "pr_star", "pphi"),
                                         first.tolist()))
                if state_by_grid != observed:
                    raise ValueError(
                        f"Public first state changed at grid {step} for row {index}")
                public_first_states[str(step)] = state_by_grid
            rows.append({
                "row": row, "same_state": {"r": state["r"], "pphi": state["pphi"],
                                           "omega": frozen["native_start_Momega"]},
                "source_flux_over_nu": source_flux,
                "wheel_PA_flux_over_nu": wheel_flux,
                "native_source_calibrated_flux_over_nu": native_flux,
                "source_mode22_same_state": [source_mode.real, source_mode.imag],
                "native_mode22_same_state": [native_mode.real, native_mode.imag],
                "native_minus_source_mode22_complex_relative": (
                    abs(native_mode-source_mode)/abs(source_mode)),
                "source_mode22_Hreal_over_M": mode_values[2],
                "native_mode22_Hreal_over_M": native_hreal,
                "source_mode_state_flux_over_nu": mode_values[3],
                "native_mode_state_flux_over_nu": native_mode_flux,
                "native_minus_source_mode_state_flux_relative": (
                    native_mode_flux/mode_values[3]-1.),
                "mode_state_flux_components": mode_flux,
                "source_placement_flux_over_nu": source_placement_flux,
                "source_placement_minus_source_flux_relative": (
                    source_placement_flux/mode_values[3]-1.),
                "source_placement_flux_components": source_placement_mode_flux,
                "source_H_real_over_nu": out[2], "source_dEdr": out[3],
                "source_dHdpr_at_0p001": out[4],
                "source_hybrid_reconstructed_pr_star": out[5],
                "source_dHdpphi": out[6], "source_d2Hdr2": out[7],
                "source_d2Hdrdpphi": out[8],
                "source_legacy_initial": initial["legacy"],
                "source_optimized_initial": initial["optimized"],
                "source_O0_legacy_initial": unoptimized["legacy"],
                "source_O0_optimized_initial": unoptimized["optimized"],
                "source_O2_fast_math_legacy_initial": fast_math["legacy"],
                "source_O2_fast_math_optimized_initial": fast_math["optimized"],
                "source_full_generator_initial": full_initial,
                "compiled_public_initial": observed,
                "compiled_public_first_states_by_step_M": public_first_states,
                "legacy_minus_compiled_public_pr_star": (
                    initial["legacy"]["pr_star"]-observed["pr_star"]),
                "optimized_minus_compiled_public_pr_star": (
                    initial["optimized"]["pr_star"]-observed["pr_star"]),
                "native_minus_source_flux_relative": native_flux/source_flux-1.,
                "wheel_minus_source_flux_relative": wheel_flux/source_flux-1.,
            })
        mode_trajectory = []
        if [item["name"] for item in trajectory["rows"]] != trajectory_spec["row_names"]:
            raise ValueError("Frozen mode-trajectory rows changed")
        for group in trajectory["rows"]:
            row = group["physical"]
            mass = row["mass1"]+row["mass2"]
            points = []
            for point in group["points"]:
                state = point["state"]
                values = (ctypes.c_double*39)()
                rc = mode_function(
                    row["mass1"]/mass, row["mass2"]/mass,
                    row["chi1"], row["chi2"], state["r"], state["phi"],
                    state["pr_star"], state["pphi"],
                    point["frequency_Momega"], values)
                if rc:
                    raise RuntimeError(
                        f"Pinned source raw mode failed at {group['name']} "
                        f"index {point['index']}: {rc}")
                source_mode = complex(values[0], values[1])
                native_mode, native_hreal, native_flux, native_components = _native_mode_and_flux(
                    row, state["r"], state["phi"], state["pr_star"],
                    state["pphi"], point["frequency_Momega"],
                    calibration["flux_choice"], plan["native_mode_choices"])
                if (abs(native_mode-source_mode)/abs(source_mode)
                        > plan["native_source_mode_max_relative_difference"]
                        or abs(native_hreal-values[2])
                        > plan["native_source_Hreal_max_absolute_difference"]):
                    raise ValueError(
                        f"Native/source raw mode mismatch at {group['name']} "
                        f"index {point['index']}")
                mode_flux = _mode_flux_comparison(
                    values, native_components,
                    plan["source_flux_mode_sum_max_relative_difference"])
                _, _, source_placement_flux, source_placement_components = (
                    _native_mode_and_flux(
                        row, state["r"], state["phi"], state["pr_star"],
                        state["pphi"], point["frequency_Momega"],
                        source_choice, plan["native_mode_choices"]))
                if (abs(source_placement_flux/values[3]-1.)
                        > plan["native_source_placement_flux_max_relative_difference"]):
                    raise ValueError(
                        f"Source-placement flux mismatch at {group['name']} "
                        f"index {point['index']}")
                source_placement_mode_flux = _mode_flux_comparison(
                    values, source_placement_components,
                    plan["source_flux_mode_sum_max_relative_difference"])
                points.append({
                    "index": point["index"],
                    "frequency_Momega": point["frequency_Momega"],
                    "source_mode22": [source_mode.real, source_mode.imag],
                    "native_mode22": [native_mode.real, native_mode.imag],
                    "complex_relative_difference": (
                        abs(native_mode-source_mode)/abs(source_mode)),
                    "source_Hreal_over_M": values[2],
                    "native_Hreal_over_M": native_hreal,
                    "source_flux_over_nu": values[3],
                    "native_flux_over_nu": native_flux,
                    "native_minus_source_flux_relative": native_flux/values[3]-1.,
                    "mode_flux_components": mode_flux,
                    "source_placement_flux_over_nu": source_placement_flux,
                    "source_placement_minus_source_flux_relative": (
                        source_placement_flux/values[3]-1.),
                    "source_placement_flux_components": source_placement_mode_flux,
                })
            mode_trajectory.append({"name": group["name"], "points": points})
    report = {
        "schema": "diffgw.seobnrv4.source-flux-initial.v6",
        "status": "diagnostic_complete", "release_qualified": False,
        "whole_waveform_parity_validated": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "adapter_sha256": adapter_spec["sha256"],
        "full_generator_adapter_sha256": full_adapter_spec["sha256"],
        "pinned_source_tag_object": plan["pinned_source"]["tag_object"],
        "pinned_source_commit": plan["pinned_source"]["peeled_commit"],
        "pinned_wheel_lib_sha256": plan["pinned_wheel"]["liblalsimulation_sha256"],
        "pinned_wheel_vcs_id": plan["pinned_wheel"]["vcs_id"],
        "source_placement_flux_choice": source_choice,
        "rows": rows,
        "mode_trajectory": mode_trajectory,
        "limitations": [
            "The full pinned-source generator and direct source IC agree exactly at the wheel's own clean VCS commit, but differ from its compiled first state; the build or numerical cause remains open.",
            "Fast-math is a compiler-sensitivity control, not evidence of the wheel's actual compiler flags.",
            "The private source raw mode is a same-state source component, not a callable wheel raw-mode comparison.",
            "Direct flux parity covers ten fixed states; it does not establish evolved waveform or compiled-wheel flux parity.",
            "Only two boundary rows and same-state components are checked; no full waveform qualification.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": report["status"], "rows": len(rows),
                      "evidence": str(args.json_out.resolve())}))


if __name__ == "__main__":
    main()
