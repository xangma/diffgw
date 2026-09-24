"""Compare pinned-wheel public SEOBNRv4 NQC fits through C and Python."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import numpy as np

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/nqc_public_c_binding_plan.json"
PROBE = q.ROOT / "dev/native_rom/reference/seobnrv4_public_c_probe.c"


def compile_probe(headers: Path, binary: dict, output: Path) -> list[str]:
    if sys.platform != "darwin":
        raise RuntimeError("the pinned wheel C-link recipe is currently macOS-only")
    paths = [Path(path) for path in binary["binaries"]]
    libraries = []
    for name in ("liblalsimulation.37.dylib", "liblal.20.dylib"):
        matches = [path for path in paths if path.name == name]
        if len(matches) != 1:
            raise ValueError(f"missing or ambiguous pinned {name}")
        libraries.append(matches[0])
    command = ["clang", "-O2", "-Wl,-headerpad_max_install_names", "-I",
               str(headers), "-o", str(output), str(PROBE),
               *(str(path) for path in libraries)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    replacements = []
    for path in libraries:
        replacements.extend(("-change", f"/DLC/lal/.dylibs/{path.name}", str(path)))
    subprocess.run(["install_name_tool", *replacements, str(output)], check=True,
                   capture_output=True, text=True)
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.nqc-public-c-binding-plan.v1"
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]
            or plan["output_step_M"] != 0.125
            or plan["max_abs_coefficient_difference"] != 1e-9):
        parser.error("frozen C-binding plan differs")
    coefficient_path = q.ROOT / plan["coefficient_evidence"]
    if q.sha256(coefficient_path) != plan["coefficient_evidence_sha256"]:
        parser.error("frozen public coefficients differ")
    prior = json.loads(coefficient_path.read_text())
    binary = prior["lal_binary"]
    if (q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary
            or [row["name"] for row in prior["rows"]] != plan["rows"]):
        parser.error("pinned binary or row order differs")
    report = dict(
        schema="diffgw.seobnrv4.nqc-public-c-binding.v1", status="running",
        scope={
            **q.SCOPE,
            "description": "Pinned public C/Python NQC binding control only.",
            "known_limitations": [
                *q.SCOPE["known_limitations"],
                "Both calls use the wheel's own evolution; native matching inputs remain distinct.",
                "This does not call the wheel's private NQC routine on native samples.",
            ],
        },
        release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), probe_sha256=q.sha256(PROBE),
        coefficient_evidence_sha256=q.sha256(coefficient_path),
        lal_binary=binary, build_command=None, rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-public-c-") as directory:
            executable = Path(directory) / "public_v4_probe"
            report["build_command"] = compile_probe(args.headers, binary, executable)
            for row in prior["rows"]:
                physical = row["row"]
                command = [str(executable), *(str(physical[key]) for key in (
                    "mass1", "mass2", "chi1", "chi2", "omega_start")),
                    str(plan["output_step_M"])]
                output = subprocess.run(command, check=True, capture_output=True, text=True)
                c_coefficients = np.fromstring(output.stdout, sep=" ")
                python_coefficients = np.asarray(row["public_coefficients"], dtype=float)
                if c_coefficients.shape != (5,) or not np.isfinite(c_coefficients).all():
                    raise ValueError(f"{row['name']} public C output is not five finite coefficients")
                difference = c_coefficients - python_coefficients
                max_abs = float(np.max(np.abs(difference)))
                if max_abs > plan["max_abs_coefficient_difference"]:
                    raise ValueError(f"{row['name']} public C/Python coefficients differ: {max_abs}")
                report["rows"].append(dict(
                    name=row["name"], row=physical,
                    c_coefficients=c_coefficients.tolist(),
                    python_coefficients=python_coefficients.tolist(),
                    c_minus_python=difference.tolist(), max_abs_difference=max_abs,
                ))
                q.write_evidence(args.json_out, report)
                print(f"row={row['name']}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
