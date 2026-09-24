"""Isolate two pinned-source Hamiltonian conventions on frozen same-state rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import torch

from dev.native_rom.eob.torch.historical_hamiltonian import HistoricalLogDerivative
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-24/hamiltonian_source_conventions_plan.json"
COEFFICIENT = re.compile(
    r"static const REAL8 coeff([0-3])([0-3])(K|dSO|dSS)\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+));"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if sha256(data) != expected:
        raise ValueError(f"Frozen input changed: {path}")
    return data


def source_coefficients(repo: Path, plan: dict) -> dict:
    tag = plan["tag_object"]
    def git(*args: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              capture_output=True).stdout
    if (git("rev-parse", plan["source_repo_tag"]).decode().strip() != tag
            or git("cat-file", "-t", tag).decode().strip() != "tag"
            or git("rev-parse", f"{tag}^{{commit}}").decode().strip()
            != plan["peeled_commit"]):
        raise ValueError("Pinned annotated source tag changed")
    data = git("show", f"{tag}:{plan['source_coefficient_header']}")
    if sha256(data) != plan["source_coefficient_header_sha256"]:
        raise ValueError("Pinned source coefficient header changed")
    coefficients = {
        (match.group(3), int(match.group(1)), int(match.group(2))):
        float(match.group(4)) for match in COEFFICIENT.finditer(data.decode())
    }
    if (len(coefficients) != plan["expected_coefficient_count"]
            or len({key[0] for key in coefficients}) != 3):
        raise ValueError("Incomplete source coefficient grid")
    return coefficients


def models(nu: float, chi1: float, chi2: float, coefficients: dict) -> dict:
    class RoundedCalibration:
        def calibration_parameters(self, r):
            n = self._parameter(self.nu, r)
            a, _, _ = self.spin_combinations(r)
            chi = a / (1.0 - 2.0*n)
            return tuple(sum(value*n**i*chi**j for (name, i, j), value
                             in coefficients.items() if name == target)
                         for target in ("K", "dSO", "dSS"))

    class RoundedPaper(RoundedCalibration, SEOBNRv4Hamiltonian):
        pass

    class RoundedHistorical(RoundedCalibration, HistoricalLogDerivative):
        pass

    args = tuple(torch.tensor(value, dtype=torch.float64)
                 for value in (nu, chi1, chi2))
    return {
        "published": SEOBNRv4Hamiltonian(*args),
        "source_calibration_only": RoundedPaper(*args),
        "historical_derivative_only": HistoricalLogDerivative(*args),
        "both_source_conventions": RoundedHistorical(*args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if plan["schema"] != "diffgw.seobnrv4.hamiltonian-source-conventions-plan.v1":
        raise ValueError("Unexpected plan schema")
    prior = json.loads(checked(ROOT / plan["same_state_evidence"],
                               plan["same_state_evidence_sha256"]))
    for key, hash_key in (("native_hamiltonian", "native_hamiltonian_sha256"),
                          ("historical_derivative_variant",
                           "historical_derivative_variant_sha256")):
        checked(ROOT / plan[key], plan[hash_key])
    coefficients = source_coefficients(args.source_root, plan)
    if (prior["status"] != "diagnostic_complete" or prior["release_qualified"]
            or sum(len(row["points"]) for row in prior["rows"])
            != plan["expected_states"]):
        raise ValueError("Frozen same-state evidence changed scope")

    rows = []
    with torch.no_grad():
        for row in prior["rows"]:
            physical = row["physical"]
            total_mass = physical["mass1"] + physical["mass2"]
            nu = physical["mass1"]*physical["mass2"]/total_mass**2
            variants = models(nu, physical["chi1"], physical["chi2"], coefficients)
            points = []
            for point in row["points"]:
                state = point["state"]
                inputs = [torch.tensor(state[key], dtype=torch.float64)
                          for key in ("r", "pr_star", "pphi")]
                source = point["wheel_H_real_over_nu"]
                values = {name: model.real_hamiltonian(*inputs).item()/nu
                          for name, model in variants.items()}
                if (abs(values["published"] - point["native_H_real_over_nu"])
                        > plan["rounding_replay_absolute_tolerance"]):
                    raise ValueError("Published baseline failed replay")
                calibrations = variants["both_source_conventions"].calibration_parameters(
                    inputs[0])
                if any(abs(value.item() - point["source_calibration"][name]) >
                       plan["rounding_replay_absolute_tolerance"]
                       for value, name in zip(calibrations, ("K", "dSO", "dSS"))):
                    raise ValueError("Parsed source calibration failed replay")
                residuals = {name: value - source for name, value in values.items()}
                if abs(residuals["both_source_conventions"]) > plan[
                        "source_convention_absolute_tolerance"]:
                    raise ValueError("Combined source convention missed pinned wheel")
                points.append({"index": point["index"], "state": state,
                               "wheel_H_real_over_nu": source,
                               "native_H_real_over_nu": values,
                               "native_minus_wheel": residuals})
            rows.append({"name": row["name"], "physical": physical,
                         "points": points})
    names = list(rows[0]["points"][0]["native_minus_wheel"])
    maxima = {name: max(abs(point["native_minus_wheel"][name])
                        for row in rows for point in row["points"]) for name in names}
    result = {
        "schema": "diffgw.seobnrv4.hamiltonian-source-conventions.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": sha256(PLAN.read_bytes()),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "same_state_evidence_sha256": plan["same_state_evidence_sha256"],
        "source_coefficient_header_sha256": plan["source_coefficient_header_sha256"],
        "source_coefficients": {f"{name}_{i}_{j}": value
                                for (name, i, j), value in coefficients.items()},
        "max_abs_native_minus_wheel": maxima, "rows": rows,
        "scope_limit": "Eight aligned-spin states only; both source conventions are development-only and do not qualify dynamics, flux, attachment, or ROM."
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "maxima": maxima}))


if __name__ == "__main__":
    main()
