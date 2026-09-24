#!/usr/bin/env python3
"""Development-only capability gate for a compiled SEOBNRv4 Hamiltonian oracle.

This intentionally does not call an opaque C wrapper without a proven v4 setup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import lal
import lalsimulation


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-24/hamiltonian_component_capability_plan.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def git_output(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Lalsuite Git repository containing the pinned annotated tag.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    source = plan["pinned_source"]
    tag_object = source["tag_object"]
    if (
        git_output(args.source_root, "rev-parse", source["tag"]).decode().strip() != tag_object
        or git_output(args.source_root, "cat-file", "-t", tag_object).decode().strip() != "tag"
        or git_output(args.source_root, "rev-parse", f"{tag_object}^{{commit}}").decode().strip()
        != source["peeled_commit"]
    ):
        raise ValueError("Pinned source tag identity changed")
    source_hashes = source["files_sha256"]
    pinned_files = {}
    for relative, expected in source_hashes.items():
        content = git_output(args.source_root, "show", f"{tag_object}:{relative}")
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned source SHA256 mismatch for {relative}")
        pinned_files[relative] = content

    wheel = plan["pinned_wheel"]
    version = getattr(lalsimulation, "__version__", None)
    distribution = importlib.metadata.version("lalsuite")
    if (lal.__version__ != wheel["lal_version"]
            or version != wheel["lalsimulation_version"]
            or distribution != wheel["lalsuite_distribution"]):
        raise ValueError("Unexpected pinned wheel versions")

    library = Path(lal.__file__).resolve().parent / ".dylibs/liblalsimulation.37.dylib"
    extensions = list(Path(lalsimulation.__file__).resolve().parent.glob("_lalsimulation*.so"))
    if len(extensions) != 1:
        raise ValueError("Pinned Python extension is ambiguous")
    extension = extensions[0]
    require_hash(library, wheel["liblalsimulation_sha256"])
    require_hash(extension, wheel["python_extension_sha256"])
    evidence_rel = plan["frozen_public_states"]["evidence"]
    evidence_path = ROOT / evidence_rel
    require_hash(evidence_path, plan["frozen_public_states"]["sha256"])
    evidence = json.loads(evidence_path.read_text())
    grid = next(g for g in evidence["grids"] if g["step_M"] == plan["frozen_public_states"]["grid_step_M"])
    points = grid["points"]
    indices = [p["index"] for p in points]
    if indices != plan["frozen_public_states"]["indices"]:
        raise ValueError(f"Frozen state indices changed: {indices}")

    exported = subprocess.run(
        ["nm", "-gU", str(library)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    symbols = sorted(
        line.rsplit(" ", 1)[-1].removeprefix("_")
        for line in exported
        if line.strip() and line.rsplit(" ", 1)[-1].startswith("_")
    )
    relevant_symbols = [
        name for name in symbols
        if "ham" in name.lower() or "spineob" in name.lower()
    ]
    python_names = sorted(
        name for name in dir(lalsimulation)
        if "hamiltonian" in name.lower() or name.startswith("SimIMRSpinEOB")
    )
    wrapper = "XLALSimInspiralEOBPAHamiltonianWrapper"
    if wrapper not in symbols:
        raise ValueError("The pinned post-adiabatic wrapper symbol is unexpectedly absent")
    header = pinned_files["lalsimulation/lib/LALSimInspiralEOBPostAdiabatic.h"].decode()
    if "SpinEOBHCoeffs *seobCoeffs" not in header or "XLALSimInspiralEOBPAHamiltonianWrapper" not in header:
        raise ValueError("The pinned wrapper declaration changed")

    result = {
        "schema": "diffgw.seobnrv4.hamiltonian-component-capability.v1",
        "status": "capability_limited",
        "plan_sha256": sha256(PLAN),
        "script_sha256": sha256(Path(__file__)),
        "pinned_source_tag_object": tag_object,
        "pinned_source_peeled_commit": source["peeled_commit"],
        "source_sha256": source_hashes,
        "wheel": {
            "lal_version": lal.__version__,
            "lalsuite_distribution": distribution,
            "lalsimulation_version": version,
            "liblalsimulation_sha256": sha256(library),
            "python_extension_sha256": sha256(extension),
        },
        "frozen_public_states": {
            "evidence": evidence_rel,
            "evidence_sha256": sha256(evidence_path),
            "row": evidence["row"],
            "grid_step_M": grid["step_M"],
            "states": [
                {
                    "index": point["index"],
                    "time_M": point["time_M"],
                    "frequency_Momega": point["frequency_Momega"],
                    "state": point["state"],
                }
                for point in points
            ],
        },
        "python_candidates": python_names,
        "exported_binary_candidates": relevant_symbols,
        "post_adiabatic_wrapper_exported": True,
        "post_adiabatic_wrapper_v4_setup_proven": False,
        "direct_v4_component_api_found": False,
        "compiled_component_parity_performed": False,
        "historical_parity_established": False,
        "finding": (
            "The pinned wheel exposes waveform APIs and a generic post-adiabatic "
            "Hamiltonian wrapper, but no directly callable, documented SEOBNRv4 "
            "Hamiltonian component with validated coefficient/branch initialization. "
            "Calling the wrapper with guessed SpinEOBHCoeffs or interpreting sampled "
            "ODE derivatives as component values would not establish parity."
        ),
        "next_qualification_step": (
            "Provide a source-backed, independently tested C adapter that initializes the "
            "exact v4 branch and coefficients at supplied states, or a public "
            "compiled component API; then compare Hamiltonian components numerically."
        ),
        "release_qualified": False,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
