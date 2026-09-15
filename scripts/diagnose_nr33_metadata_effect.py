#!/usr/bin/env python3
"""Bounded q3 sensitivity of the frozen NR33 prediction to Lev3 metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

try:
    from scripts.fit_nr33_amplitude_pilot import independent_qnm33, read_manifest, ringdown_shape
except ModuleNotFoundError:  # direct ``python scripts/...py`` invocation
    from fit_nr33_amplitude_pilot import independent_qnm33, read_manifest, ringdown_shape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAL = ROOT / "docs/reviews/2026-09-15-independent-calibration/nr33-calibration.json"
DEFAULT_MANIFEST = ROOT / "docs/reviews/2026-09-15-independent-calibration/nr33-data-manifest.json"
DEFAULT_SAVED = ROOT / "docs/reviews/2026-09-15-independent-calibration/nr33-frozen-predictions.json"
DEFAULT_CONSISTENCY = ROOT / "docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis/metadata-consistency.json"
DEFAULT_METADATA = Path("/tmp/torchwave-nr33-metadata-audit/SXS_BBH_0030_Lev3_metadata.json")
EXPECTED_CAL_SHA256 = "f447acda86f64f1d13745c8a5d725acdf295b050a2638d1731d4b6a139e1358d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qnm_delta(base: dict, value: dict) -> dict:
    return {
        "f_ring_33_absolute": float(value["f_ring_33"] - base["f_ring_33"]),
        "f_ring_33_relative": float(value["f_ring_33"] / base["f_ring_33"] - 1),
        "f_damp_33_absolute": float(value["f_damp_33"] - base["f_damp_33"]),
        "f_damp_33_relative": float(value["f_damp_33"] / base["f_damp_33"] - 1),
    }


def prediction(f: np.ndarray, q: float, qnm: dict, calibration: dict) -> np.ndarray:
    """The prediction expression used by run_nr33_frozen_validation.py."""
    params = calibration["parameter_fit"]
    eta = q / (1 + q) ** 2
    delta = (q - 1) / (q + 1)
    x = 4 * eta - 1
    log_a = np.polynomial.polynomial.polyval(
        x, params["log_abs_a_lambda_over_delta_coefficients"])
    lam = np.polynomial.polynomial.polyval(x, params["lambda_coefficients"])
    a022 = np.pi * np.sqrt(2 * eta / 3) * (np.pi * f) ** (-7 / 6)
    out = a022 * delta * np.exp(log_a) * ringdown_shape(
        f, qnm["f_ring_33"], qnm["f_damp_33"], lam, calibration["sigma_33"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--saved-predictions", type=Path, default=DEFAULT_SAVED)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--metadata-consistency", type=Path, default=DEFAULT_CONSISTENCY)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat()

    if sha256(args.calibration) != EXPECTED_CAL_SHA256:
        raise RuntimeError("frozen calibration SHA256 mismatch")
    calibration = json.loads(args.calibration.read_text())
    manifest_cases = read_manifest(args.manifest)
    case = next(c for c in manifest_cases if c.case_id == "SXS:BBH:0030/Lev3/N3")
    saved = json.loads(args.saved_predictions.read_text())
    saved_case = next(x for x in saved["predictions"] if x["case_id"] == case.case_id)
    f = np.asarray(saved_case["frequency_Mf"], dtype=float)
    saved_y = np.asarray(saved_case["prediction_amplitude_geometric"], dtype=float)
    if f.size != saved_y.size or f.size < 8 or not np.all(np.diff(f) > 0):
        raise RuntimeError("saved q3 frequency bins are invalid")

    consistency = json.loads(args.metadata_consistency.read_text())
    audit = next(x for x in consistency["cases"]
                 if x.get("id") == "SXS:BBH:0030/Lev3/N3")
    audit_fields = audit["exact_selected_lev_metadata_fields"]
    cache = json.loads(args.metadata.read_text())
    if md5(args.metadata) != consistency["md5_verification"]["SXS:BBH:0030/Lev3"]["actual_md5"]:
        raise RuntimeError("cached Lev3 metadata MD5 mismatch")
    for key, cache_key in (("reference_mass1", "reference_mass1"),
                           ("reference_mass2", "reference_mass2"),
                           ("remnant_mass", "remnant_mass")):
        if not np.isclose(float(cache[cache_key]), float(audit_fields[key]), rtol=0, atol=1e-14):
            raise RuntimeError("metadata consistency mismatch for " + key)
    if not np.allclose(cache["remnant_dimensionless_spin"], audit_fields["remnant_spin"], rtol=0, atol=1e-14):
        raise RuntimeError("metadata consistency mismatch for remnant spin")

    # Baseline is manifest QNM metadata and manifest q. The bins are loaded from
    # the saved frozen prediction and are never recomputed for any variant.
    base_qnm = independent_qnm33(case.metadata)
    base_q = float(case.q)
    base_y = prediction(f, base_q, base_qnm, calibration)
    baseline_reproduction = {
        "max_absolute_difference": float(np.max(np.abs(base_y - saved_y))),
        "relative_l2": float(np.linalg.norm(base_y - saved_y) / np.linalg.norm(saved_y)),
        "saved_qnm": saved_case["qnm"],
        "recomputed_qnm": base_qnm,
        "q": base_q,
    }
    if baseline_reproduction["relative_l2"] > 1e-13:
        raise RuntimeError("frozen prediction baseline was not reproduced")

    exact_mass_fraction = float(cache["remnant_mass"] /
                                (cache["reference_mass1"] + cache["reference_mass2"]))
    exact_spin = float(np.linalg.norm(np.asarray(cache["remnant_dimensionless_spin"], dtype=float)))
    exact_meta = {"final_mass_fraction": exact_mass_fraction, "final_spin_magnitude": exact_spin}
    qnm_mass_only = independent_qnm33({"final_mass_fraction": exact_mass_fraction,
                                       "final_spin_magnitude": float(case.metadata["final_spin_magnitude"])})
    qnm_spin_only = independent_qnm33({"final_mass_fraction": float(case.metadata["final_mass_fraction"]),
                                       "final_spin_magnitude": exact_spin})
    qnm_exact = independent_qnm33(exact_meta)
    q_ref = float(audit_fields["q"])

    variants = [
        ("manifest_baseline", base_q, base_qnm, "manifest remnant metadata and q"),
        ("correct_Lev3_remnant_mass_only", base_q, qnm_mass_only,
         "exact Lev3 remnant_mass/(reference_mass1+reference_mass2); baseline spin"),
        ("correct_Lev3_remnant_spin_only", base_q, qnm_spin_only,
         "exact Lev3 remnant spin-vector magnitude; baseline mass fraction"),
        ("correct_Lev3_remnant_mass_and_spin", base_q, qnm_exact,
         "exact Lev3 remnant mass fraction and spin-vector magnitude; baseline q"),
        ("reference_q_only", q_ref, base_qnm, "exact Lev3 reference_mass_ratio; baseline QNM"),
        ("correct_Lev3_remnant_and_reference_q", q_ref, qnm_exact,
         "exact Lev3 remnant inputs and exact Lev3 reference_mass_ratio"),
    ]
    results = []
    for name, q, qnm, change in variants:
        y = prediction(f, q, qnm, calibration)
        results.append({"name": name, "description": change, "q": q, "qnm": qnm,
                        "qnm_shift_from_manifest": qnm_delta(base_qnm, qnm),
                        "prediction_change_relative_l2": float(np.linalg.norm(y - base_y) / np.linalg.norm(base_y)),
                        "prediction_norm_ratio_over_manifest": float(np.linalg.norm(y) / np.linalg.norm(base_y)),
                        "prediction_max_relative_change": float(np.max(np.abs(y / base_y - 1))),
                        "prediction_first_last": [float(y[0]), float(y[-1])]})

    if sha256(args.calibration) != EXPECTED_CAL_SHA256:
        raise RuntimeError("calibration changed during diagnostic")
    report = {
        "purpose": "bounded sensitivity of the frozen q3 Lev3 prediction to corrected remnant metadata; no refit and no LAL",
        "case_id": case.case_id,
        "frequency_bins": {"source": str(args.saved_predictions), "count": int(f.size),
                            "Mf_range": [float(f[0]), float(f[-1])], "sha256": sha256(args.saved_predictions)},
        "formula_source": {"script": "scripts/run_nr33_frozen_validation.py::prepare_inputs",
                            "prediction_expression": "a022*delta*exp(log_a)*ringdown_shape(f,f_ring_33,f_damp_33,lambda,sigma_33)",
                            "helper_script": str(ROOT / "scripts/fit_nr33_amplitude_pilot.py"),
                            "helper_sha256": sha256(ROOT / "scripts/fit_nr33_amplitude_pilot.py"),
                            "kerr_solver": str(ROOT / "scripts/kerr_leaver_solver.py"),
                            "kerr_solver_sha256": sha256(ROOT / "scripts/kerr_leaver_solver.py")},
        "controls": {"sigma_33": calibration["sigma_33"], "frozen_frequency_bins": True,
                     "f_rd_rule": calibration["f_rd_rule"], "fmax": calibration["controls"]["fmax"]},
        "baseline_reproduction": baseline_reproduction,
        "metadata_inputs": {
            "manifest_q": base_q,
            "manifest_final_mass_fraction": float(case.metadata["final_mass_fraction"]),
            "manifest_final_spin_magnitude": float(case.metadata["final_spin_magnitude"]),
            "exact_Lev3_reference_mass_ratio": q_ref,
            "exact_Lev3_reference_mass1": float(cache["reference_mass1"]),
            "exact_Lev3_reference_mass2": float(cache["reference_mass2"]),
            "exact_Lev3_remnant_mass": float(cache["remnant_mass"]),
            "exact_Lev3_remnant_mass_fraction": exact_mass_fraction,
            "exact_Lev3_remnant_spin_vector": cache["remnant_dimensionless_spin"],
            "exact_Lev3_remnant_spin_magnitude": exact_spin,
            "cache_path": str(args.metadata),
            "source_url": consistency["md5_verification"]["SXS:BBH:0030/Lev3"]["url"],
            "cache_sha256": sha256(args.metadata),
            "cache_md5": md5(args.metadata), "consistency_report_sha256": sha256(args.metadata_consistency)},
        "variants": results,
        "calibration_sha256": sha256(args.calibration),
        "manifest_sha256": sha256(args.manifest),
        "run": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(), "cwd": str(Path.cwd()), "python": sys.executable,
                "command": sys.argv, "script_sha256": sha256(Path(__file__))},
        "limitations": ["This is a metadata-only perturbation of the frozen analytic prediction on its saved bins.",
                         "It bounds remnant-input sensitivity and does not validate the whole fitted model or explain NR extraction errors."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"bins": int(f.size), "baseline_relative_l2": baseline_reproduction["relative_l2"],
                      "output": str(args.output), "calibration_unchanged": True}))


if __name__ == "__main__":
    main()
