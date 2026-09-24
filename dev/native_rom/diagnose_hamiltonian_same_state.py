"""Development-only pinned-wheel SEOBNRv4 Hamiltonian same-state comparator.

The adapter compiles the initializer and Hamiltonian from the pinned *tag*,
then calls the wheel's exported PA Hamiltonian wrapper with those coefficients.
The native implementation remains independent of both.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import lal
import lalsimulation
import torch

from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-24/hamiltonian_same_state_plan.json"
ADAPTER = ROOT / "dev/native_rom/reference/hamiltonian_v4_source_wheel_adapter.c"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def checked_file(path: Path, expected: str) -> str:
    actual = file_hash(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}: {actual} != {expected}")
    return actual


def source_snapshot(repo: Path, source: dict, destination: Path) -> dict:
    tag = source["tag_object"]
    if (git(repo, "rev-parse", source["tag"]).decode().strip() != tag
            or git(repo, "cat-file", "-t", tag).decode().strip() != "tag"
            or git(repo, "rev-parse", f"{tag}^{{commit}}").decode().strip()
            != source["peeled_commit"]):
        raise ValueError("Pinned annotated source tag changed")
    hashes = {}
    for relative, expected in source["files_sha256"].items():
        content = git(repo, "show", f"{tag}:{relative}")
        actual = digest(content)
        if actual != expected:
            raise ValueError(f"Pinned source changed: {relative}")
        # The C adapter includes by basename, so the pinned blob must be the
        # only matching source file on the compiler's private include path.
        target = destination / Path(relative).name
        target.write_bytes(content)
        hashes[relative] = actual
    return hashes


def check_public_headers(repo: Path, source: dict, headers: Path) -> dict:
    hashes = {}
    for relative, expected in source["public_headers_sha256"].items():
        content = git(repo, "show", f"{source['tag_object']}:{relative}")
        if digest(content) != expected:
            raise ValueError(f"Pinned public header changed: {relative}")
        checked_file(headers / "lal" / Path(relative).name, expected)
        hashes[relative] = expected
    return hashes


def compile_adapter(headers: Path, gsl_include: Path, private: Path,
                    libraries: dict[str, Path], output: Path) -> list[str]:
    command = [
        "clang", "-dynamiclib", "-O2", "-H", "-Wno-gnu-folding-constant",
        "-Wl,-headerpad_max_install_names", "-I", str(private),
        "-I", str(headers), "-I", str(gsl_include),
        "-o", str(output), str(ADAPTER),
        *(str(path) for path in libraries.values()),
    ]
    compilation = subprocess.run(command, check=True, capture_output=True, text=True)
    for name in ("LALSimIMRSpinEOBHamiltonian.c", "LALSimIMRSpinEOBHamiltonian.h",
                 "LALSimIMRSpinEOB.h", "LALSimIMREOBNRv2.h"):
        if str(private / name) not in compilation.stderr:
            raise ValueError(f"Compiler did not include pinned {name}")
    changes = []
    for name, path in libraries.items():
        changes.extend(("-change", f"/DLC/lal/.dylibs/{name}", str(path)))
    subprocess.run(["install_name_tool", *changes, str(output)],
                   check=True, capture_output=True, text=True)
    imports = subprocess.run(["nm", "-u", str(output)], check=True,
                             capture_output=True, text=True).stdout
    if "_XLALSimInspiralEOBPAHamiltonianWrapper" not in imports:
        raise ValueError("Adapter did not import the wheel Hamiltonian wrapper")
    bindings = subprocess.run(["nm", "-m", str(output)], check=True,
                              capture_output=True, text=True).stdout
    if ("_XLALSimInspiralEOBPAHamiltonianWrapper "
            "(from liblalsimulation.37)" not in bindings):
        raise ValueError("Adapter wrapper import is not bound to wheel dylib")
    linked = subprocess.run(["otool", "-L", str(output)], check=True,
                            capture_output=True, text=True).stdout
    for name, path in libraries.items():
        if str(path) not in linked:
            raise ValueError(f"Adapter did not link pinned wheel library {name}")
    return command


def frozen_points(spec: dict) -> tuple[dict, list[dict]]:
    evidence_path = ROOT / spec["evidence"]
    checked_file(evidence_path, spec["sha256"])
    evidence = json.loads(evidence_path.read_text())
    if isinstance(evidence["grids"], list):
        grid = next(g for g in evidence["grids"]
                    if g["step_M"] == spec["grid_step_M"])
        points = grid["points"]
    else:
        grid = evidence["grids"][str(spec["grid_step_M"])]
        points = grid["targets"]
    if [point["index"] for point in points] != spec["indices"]:
        raise ValueError(f"Frozen indices changed for {spec['name']}")
    return evidence["row"], points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Git repository containing lalsimulation-v6.2.1 tag")
    parser.add_argument("--headers", type=Path, required=True,
                        help="Merged LAL headers from the pinned source snapshot")
    parser.add_argument("--gsl-include", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise RuntimeError("This pinned-wheel C-link recipe is macOS-only")
    plan = json.loads(PLAN.read_text())
    if plan["schema"] != "diffgw.seobnrv4.hamiltonian-same-state-plan.v1":
        raise ValueError("Unexpected same-state plan")
    checked_file(ROOT / plan["native_hamiltonian"]["path"],
                 plan["native_hamiltonian"]["sha256"])
    wheel = plan["pinned_wheel"]
    if (importlib.metadata.version("lalsuite") != wheel["lalsuite_distribution"]
            or lal.__version__ != wheel["lal_version"]
            or lalsimulation.__version__ != wheel["lalsimulation_version"]):
        raise ValueError("Pinned wheel version changed")
    library_dir = Path(lal.__file__).resolve().parent / ".dylibs"
    library = library_dir / "liblalsimulation.37.dylib"
    extensions = list(Path(lalsimulation.__file__).resolve().parent.glob(
        "_lalsimulation*.so"))
    if len(extensions) != 1:
        raise ValueError(f"Expected one pinned lalsimulation extension, got {extensions}")
    extension = extensions[0]
    checked_file(library, wheel["liblalsimulation_sha256"])
    checked_file(extension, wheel["python_extension_sha256"])
    exports = subprocess.run(["nm", "-gU", str(library)], check=True,
                             capture_output=True, text=True).stdout
    if "_XLALSimInspiralEOBPAHamiltonianWrapper" not in exports:
        raise ValueError("Wheel Hamiltonian wrapper symbol absent")
    libraries = {name: library_dir / name for name in (
        "liblalsimulation.37.dylib", "liblal.20.dylib",
        "libgsl.28.dylib", "libgslcblas.0.dylib")}
    for name, path in libraries.items():
        checked_file(path, wheel["libraries_sha256"][name])

    with tempfile.TemporaryDirectory(prefix="diffgw-v4-hamiltonian-") as tmp:
        private = Path(tmp) / "private"
        private.mkdir()
        source_hashes = source_snapshot(args.source_root, plan["pinned_source"], private)
        public_header_hashes = check_public_headers(
            args.source_root, plan["pinned_source"], args.headers)
        adapter = Path(tmp) / "v4_hamiltonian_adapter.dylib"
        command = compile_adapter(args.headers, args.gsl_include,
                                  private, libraries, adapter)
        function = ctypes.CDLL(str(adapter)).diffgw_v4_hamiltonian_same_state
        function.argtypes = [ctypes.c_double] * 7 + [ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        def scalar(value: float) -> torch.Tensor:
            return torch.tensor(value, dtype=torch.float64)
        rows = []
        with torch.no_grad():
            for spec in plan["frozen_states"]:
                physical, points = frozen_points(spec)
                mass = physical["mass1"] + physical["mass2"]
                x1, x2 = physical["mass1"] / mass, physical["mass2"] / mass
                nu = x1 * x2
                native = SEOBNRv4Hamiltonian(
                    scalar(nu), scalar(physical["chi1"]), scalar(physical["chi2"]))
                records = []
                for point in points:
                    state = point["state"]
                    out = (ctypes.c_double * 5)()
                    rc = function(x1, x2, physical["chi1"], physical["chi2"],
                                  state["r"], state["pr_star"], state["pphi"], out)
                    if rc != 0:
                        raise RuntimeError(f"{spec['name']} index {point['index']}: C rc={rc}")
                    source_h, wheel_h, k, d_so, d_ss = out
                    difference = wheel_h - source_h
                    if abs(difference) > plan["source_wheel_max_absolute_H_over_nu_difference"]:
                        raise ValueError(f"Source/wheel mismatch at {spec['name']} {point['index']}")
                    radius = scalar(state["r"])
                    native_h = native.real_hamiltonian(
                        radius, scalar(state["pr_star"]), scalar(state["pphi"])).item()/nu
                    nk, nso, nss = native.calibration_parameters(radius)
                    records.append({
                        "index": point["index"], "time_M": point["time_M"],
                        "frequency_Momega": point["frequency_Momega"],
                        "state": state,
                        "source_H_real_over_nu": source_h,
                        "wheel_H_real_over_nu": wheel_h,
                        "native_H_real_over_nu": native_h,
                        "wheel_minus_source": difference,
                        "native_minus_wheel": native_h - wheel_h,
                        "native_minus_wheel_relative": (native_h - wheel_h)/wheel_h,
                        "source_calibration": {"K": k, "dSO": d_so, "dSS": d_ss},
                        "native_calibration": {"K": nk.item(), "dSO": nso.item(),
                                               "dSS": nss.item()},
                    })
                rows.append({
                    "name": spec["name"], "physical": physical,
                    "frozen_evidence": spec["evidence"],
                    "frozen_evidence_sha256": spec["sha256"],
                    "max_abs_wheel_minus_source": max(abs(p["wheel_minus_source"])
                                                       for p in records),
                    "max_abs_native_minus_wheel": max(abs(p["native_minus_wheel"])
                                                       for p in records),
                    "max_abs_relative_native_minus_wheel": max(
                        abs(p["native_minus_wheel_relative"]) for p in records),
                    "points": records,
                })

    result = {
        "schema": "diffgw.seobnrv4.hamiltonian-same-state.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": file_hash(PLAN), "script_sha256": file_hash(Path(__file__)),
        "adapter_sha256": file_hash(ADAPTER),
        "pinned_source_tag_object": plan["pinned_source"]["tag_object"],
        "pinned_source_peeled_commit": plan["pinned_source"]["peeled_commit"],
        "source_files_sha256": source_hashes,
        "public_headers_sha256": public_header_hashes,
        "wheel": wheel,
        "wheel_exported_symbol": "XLALSimInspiralEOBPAHamiltonianWrapper",
        "wheel_symbol_binding": "liblalsimulation.37",
        "source_initializer": "XLALSimIMRCalculateSpinEOBHCoeffs(..., 4)",
        "compile_command": command,
        "compiler": subprocess.run(["clang", "--version"], check=True,
                                   capture_output=True, text=True).stdout.splitlines()[0],
        "header_root": str(args.headers), "gsl_include": str(args.gsl_include),
        "rows": rows,
        "scope_limit": (
            "The wheel wrapper is called with coefficients initialized by the exact pinned "
            "source v4 initializer. The wheel's private initializer is not exported; "
            "the source/wheel agreement validates this adapter at the tested states, "
            "not all Hamiltonian branches or the full waveform."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "max_abs_native_minus_wheel": max(
                          row["max_abs_native_minus_wheel"] for row in rows)}))


if __name__ == "__main__":
    main()
