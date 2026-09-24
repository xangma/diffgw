"""Reproduce pinned-source v4 flux and initial-condition component checks.

This development diagnostic never supplies LAL code to the native generator.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
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


ROOT = q.ROOT
PLAN = ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json"


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


def _libraries(spec: dict) -> dict[str, Path]:
    if (importlib.metadata.version("lalsuite") != spec["lalsuite_distribution"]
            or lal.__version__ != spec["lal_version"]
            or lalsimulation.__version__ != spec["lalsimulation_version"]):
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
             required: dict[str, str], optimization: str = "-O2") -> None:
    if optimization not in ("-O0", "-O2"):
        raise ValueError("Unsupported compiler comparison")
    command = [
        "clang", "-dynamiclib", optimization, "-H", "-Wno-gnu-folding-constant",
        "-Wl,-headerpad_max_install_names",
        "-I", str(private / "lalsimulation/lib"),
        "-I", str(private / "lal/lib/factories"),
        "-I", str(headers), "-I", str(gsl),
        "-o", str(output), str(adapter),
        *(str(path) for path in libraries.values()),
    ]
    build = subprocess.run(command, check=True, capture_output=True, text=True)
    for relative in required:
        if str(private / relative) not in build.stderr:
            raise ValueError(f"Compiler did not include pinned {relative}")
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
    if ("_XLALSimInspiralEOBPAFluxWrapper "
            "(from liblalsimulation.37)" not in bindings):
        raise ValueError("PA flux wrapper was not imported from pinned wheel")


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
    if (plan["schema"] != "diffgw.seobnrv4.source-flux-initial-plan.v1"
            or plan["release_qualified"] is not False):
        raise ValueError("Unexpected diagnostic plan")
    adapter_spec = plan["adapter"]
    adapter = ROOT / adapter_spec["path"]
    checked_file(adapter, adapter_spec["sha256"])
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
                 output, plan["pinned_source"]["files_sha256"])
        library = ctypes.CDLL(str(output))
        function = library.diffgw_v4_flux_same_state
        function.argtypes = [ctypes.c_double]*7 + [ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        initial_function = library.diffgw_v4_initial_conditions
        initial_function.argtypes = [ctypes.c_double]*5 + [ctypes.c_int,
                                                           ctypes.POINTER(ctypes.c_double)]
        initial_function.restype = ctypes.c_int
        unoptimized_output = Path(tmp) / "adapter_O0.dylib"
        _compile(private, args.headers, args.gsl_include, libraries, adapter,
                 unoptimized_output, plan["pinned_source"]["files_sha256"], "-O0")
        unoptimized_library = ctypes.CDLL(str(unoptimized_output))
        unoptimized_initial = unoptimized_library.diffgw_v4_initial_conditions
        unoptimized_initial.argtypes = initial_function.argtypes
        unoptimized_initial.restype = ctypes.c_int
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
            for name, optimized in (("legacy", 0), ("optimized", 1)):
                values = (ctypes.c_double*3)()
                rc = unoptimized_initial(row["mass1"], row["mass2"],
                                         row["chi1"], row["chi2"],
                                         frozen["native_start_Momega"], optimized, values)
                if rc:
                    raise RuntimeError(
                        f"Pinned source {name} O0 IC failed at row {index}: {rc}")
                unoptimized[name] = dict(zip(("r", "pr_star", "pphi"), values))
            source_flux, wheel_flux = out[0], out[1]
            if abs(wheel_flux/source_flux-1.) > plan["source_wheel_max_relative_flux_difference"]:
                raise ValueError(f"Source/wheel flux mismatch at row {index}")
            native_flux = _native_flux(row, state["r"], state["pphi"],
                                       frozen["native_start_Momega"],
                                       calibration["flux_choice"])
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
                "source_H_real_over_nu": out[2], "source_dEdr": out[3],
                "source_dHdpr_at_0p001": out[4],
                "source_hybrid_reconstructed_pr_star": out[5],
                "source_dHdpphi": out[6], "source_d2Hdr2": out[7],
                "source_d2Hdrdpphi": out[8],
                "source_legacy_initial": initial["legacy"],
                "source_optimized_initial": initial["optimized"],
                "source_O0_legacy_initial": unoptimized["legacy"],
                "source_O0_optimized_initial": unoptimized["optimized"],
                "compiled_public_initial": observed,
                "compiled_public_first_states_by_step_M": public_first_states,
                "legacy_minus_compiled_public_pr_star": (
                    initial["legacy"]["pr_star"]-observed["pr_star"]),
                "optimized_minus_compiled_public_pr_star": (
                    initial["optimized"]["pr_star"]-observed["pr_star"]),
                "native_minus_source_flux_relative": native_flux/source_flux-1.,
                "wheel_minus_source_flux_relative": wheel_flux/source_flux-1.,
            })
    report = {
        "schema": "diffgw.seobnrv4.source-flux-initial.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "whole_waveform_parity_validated": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "adapter_sha256": adapter_spec["sha256"],
        "pinned_source_tag_object": plan["pinned_source"]["tag_object"],
        "pinned_wheel_lib_sha256": plan["pinned_wheel"]["liblalsimulation_sha256"],
        "rows": rows,
        "limitations": [
            "Direct source and compiled wheel initial conditions do not agree at the radial-momentum level; the cause remains open.",
            "Only two boundary rows and same-state components are checked; no full waveform qualification.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": report["status"], "rows": len(rows),
                      "evidence": str(args.json_out.resolve())}))


if __name__ == "__main__":
    main()
