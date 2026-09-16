#!/usr/bin/env python3
"""Validate a frozen NR33 calibration on explicitly admitted external holdouts.

This adapter is deliberately separate from the frozen-fit holdout runner.  It
accepts only a quality-screened, holdout-only manifest, checks that its SXS
simulations are disjoint from the frozen fit, and reuses the frozen numerical
preprocessing and coefficients.  It never calls a fitter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.fit_nr33_amplitude_pilot import (
    Case, FFTControls, frequency_amplitude, independent_qnm33, load_case,
    ringdown_shape,
)
from scripts.validate_nr33_amplitude_pilot import sha256_file, validate_frozen_predictions


FROZEN_CALIBRATION_SHA256 = "ade377c62a57a5d9cff513e34f22b0fe43677177a0c1577a9a4af5ad3b8aa561"
FROZEN_PROTOCOL_SHA256 = "e337fac253a230da3250593d0edfad8bcd2a9cd7540d0e6f811010d9c8c1f67a"
FROZEN_POLICY_SHA256 = "1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _external_cases(path: Path) -> list[Case]:
    raw = _json(path)
    entries = raw.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("external manifest must contain a non-empty cases list")
    cases: list[Case] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"external case {index} is not an object")
        if "id" not in entry or "path" not in entry or "q" not in entry:
            raise ValueError(f"external case {index} needs explicit id, path, and q")
        metadata = dict(entry.get("metadata", {}))
        for key in (
            "f_ring_33", "f_damp_33", "f_ring", "f_damp", "final_spin",
            "finalspin", "final_spin_magnitude", "finalspin_magnitude",
            "remnant_spin", "chi_final", "final_mass_fraction", "finalmassfraction",
            "remnant_mass_fraction", "relaxation_time", "peak_time", "sxs_id",
            "split", "npz_sha256", "extrapolation_order",
            "target_q", "candidate_rank",
        ):
            if key in entry:
                metadata[key] = entry[key]
        path_value = Path(str(entry["path"]))
        if not path_value.is_absolute():
            path_value = path.parent / path_value
        q = float(entry["q"])
        if not np.isfinite(q) or q <= 1:
            raise ValueError(f"external case {index} must have finite q>1")
        cases.append(Case(str(entry["id"]), str(path_value), q, metadata))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("external case IDs must be unique")
    return cases


def _sxs_id(case: Case) -> str:
    value = case.metadata.get("sxs_id")
    if not value:
        raise ValueError(f"external case {case.case_id} lacks explicit sxs_id")
    return str(value)


def _quality_admission(quality_path: Path, manifest_path: Path, cases: list[Case]) -> dict[str, Any]:
    quality = _json(quality_path)
    if quality.get("policy_sha256") != FROZEN_POLICY_SHA256:
        raise ValueError("quality report does not use the pinned quality policy")
    if quality.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256:
        raise ValueError("quality report does not use the pinned expanded protocol")
    manifest_sha = sha256_file(manifest_path)
    input_manifest_sha = quality.get("input_manifest_sha256")
    if not isinstance(input_manifest_sha, str) or len(input_manifest_sha) != 64:
        raise ValueError("quality report lacks the full input-manifest hash")
    selected = quality.get("selected_primary_case_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError("quality report needs non-empty selected_primary_case_ids")
    ids = [case.case_id for case in cases]
    if selected != ids:
        raise ValueError("quality selected IDs do not exactly match external manifest")
    qualified = quality.get("qualified_manifest")
    if not isinstance(qualified, Mapping) or qualified.get("sha256") != manifest_sha:
        raise ValueError("quality qualified-manifest hash does not match external manifest")
    qualified_payload = _json(manifest_path)
    screening = qualified_payload.get("screening")
    if not isinstance(screening, Mapping) or screening.get("manifest_sha256") != input_manifest_sha:
        raise ValueError("qualified manifest does not bind to the screened input manifest")
    if screening.get("policy_sha256") != FROZEN_POLICY_SHA256:
        raise ValueError("qualified manifest does not use the pinned quality policy")
    if screening.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256:
        raise ValueError("qualified manifest does not use the pinned expanded protocol")
    if screening.get("selected_primary_case_ids") != ids:
        raise ValueError("qualified manifest selected IDs do not exactly match external manifest")
    if int(qualified.get("case_count", len(ids))) != len(ids):
        raise ValueError("quality qualified-manifest case count mismatch")
    simulations = quality.get("simulations")
    if not isinstance(simulations, list):
        raise ValueError("quality report needs simulations")
    by_id = {str(row.get("primary_case_id")): row for row in simulations if isinstance(row, Mapping)}
    for case_id in ids:
        row = by_id.get(case_id)
        if row is None or row.get("passed") is not True or row.get("evaluation_status") not in {None, "passed"}:
            raise ValueError(f"external case did not individually pass quality gates: {case_id}")
    # An external screen may report overall_pass=false only because the full
    # requested target grid is absent.  That aggregate gap is admissible here; the
    # selected rows and their individual gates remain mandatory.
    if quality.get("overall_pass") is not True:
        counts = quality.get("counts", {})
        required = counts.get("required_targets") if isinstance(counts, Mapping) else None
        selected_count = counts.get("selected") if isinstance(counts, Mapping) else None
        if not (isinstance(required, int) and isinstance(selected_count, int)
                and selected_count == len(ids) and required > selected_count):
            raise ValueError("external quality report did not pass")
    return quality


def _check_frozen_inputs(calibration: Path, fit: Mapping[str, Any], expected_sha256: str) -> str:
    if expected_sha256 != FROZEN_CALIBRATION_SHA256:
        raise ValueError("requested calibration SHA is not the pinned v3 freeze")
    actual = sha256_file(calibration)
    if actual != FROZEN_CALIBRATION_SHA256:
        raise ValueError("calibration differs from pinned v3 frozen SHA256")
    provenance = fit.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("frozen calibration lacks provenance")
    if provenance.get("quality_policy_sha256") != FROZEN_POLICY_SHA256:
        raise ValueError("frozen calibration quality policy provenance changed")
    manifest_value = provenance.get("manifest_path")
    manifest_digest = provenance.get("manifest_sha256")
    if not isinstance(manifest_value, str) or not isinstance(manifest_digest, str):
        raise ValueError("frozen calibration lacks original fit manifest provenance")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_file() or manifest_digest != sha256_file(manifest_path):
        raise ValueError("original fit manifest differs from frozen provenance hash")
    fitter = Path(__file__).with_name("fit_nr33_amplitude_pilot.py")
    fitter_digest = provenance.get("fitter_script_sha256")
    if not isinstance(fitter_digest, str) or sha256_file(fitter) != fitter_digest:
        raise ValueError("frozen numerical fitter source changed")
    qnm_value = provenance.get("qnm_helper_path")
    qnm_digest = provenance.get("qnm_helper_sha256")
    qnm_path = Path(qnm_value) if isinstance(qnm_value, str) else Path("")
    if not qnm_path.is_file() or not isinstance(qnm_digest, str) or sha256_file(qnm_path) != qnm_digest:
        raise ValueError("frozen Kerr helper source changed")
    return actual


def _check_external(cases: list[Case], manifest_path: Path, fit: Mapping[str, Any],
                    protocol: Mapping[str, Any]) -> None:
    split = fit.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("frozen calibration lacks split")
    frozen_ids = set(str(x) for x in split.get("train_case_ids", [])) | set(str(x) for x in split.get("holdout_case_ids", []))
    frozen_sxs = {item.split("/", 1)[0] for item in frozen_ids}
    excluded_sxs = {str(x) for x in protocol.get("excluded_sxs_ids", [])}
    target_grid = [float(x) for x in protocol.get("target_mass_ratios", [])]
    max_q_deviation = float(protocol.get("max_relative_q_deviation", 0.005))
    if not excluded_sxs or not target_grid:
        raise ValueError("expanded protocol lacks exclusion or target declarations")
    seen_sxs: set[str] = set()
    for case in cases:
        if str(case.metadata.get("split", "")).lower() != "holdout":
            raise ValueError(f"external case is not holdout: {case.case_id}")
        extrapolation = case.metadata.get("extrapolation_order")
        if isinstance(extrapolation, bool) or not isinstance(extrapolation, (int, float)) or int(extrapolation) != extrapolation or int(extrapolation) != 3:
            raise ValueError(f"external case must be N3: {case.case_id}")
        if not case.case_id.endswith("/N3"):
            raise ValueError(f"external case ID must identify N3: {case.case_id}")
        sxs = _sxs_id(case)
        if sxs in excluded_sxs:
            raise ValueError(f"external SXS simulation is excluded by protocol: {sxs}")
        if sxs in frozen_sxs:
            raise ValueError(f"external SXS simulation overlaps frozen fit: {sxs}")
        if sxs in seen_sxs:
            raise ValueError(f"external manifest repeats SXS simulation: {sxs}")
        seen_sxs.add(sxs)
        target_q = case.metadata.get("target_q")
        if target_q is None:
            raise ValueError(f"external case lacks explicit target_q: {case.case_id}")
        target_q = float(target_q)
        if not np.isfinite(target_q) or not any(abs(target_q - target) <= 1e-12 for target in target_grid):
            raise ValueError(f"external case target_q is outside protocol grid: {case.case_id}")
        if abs(case.q / target_q - 1.0) > max_q_deviation:
            raise ValueError(f"external case q is outside declared target tolerance: {case.case_id}")
        try:
            rank = case.metadata["candidate_rank"]
            if isinstance(rank, bool) or int(rank) != rank or int(rank) < 1:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"external case lacks a positive candidate_rank: {case.case_id}") from exc
        expected = case.metadata.get("npz_sha256")
        if not expected:
            raise ValueError(f"external case lacks NPZ SHA256: {case.case_id}")
        if not Path(case.path).is_file():
            raise ValueError(f"external NPZ does not exist: {case.path}")
        if sha256_file(Path(case.path)) != str(expected).lower():
            raise ValueError(f"external NPZ SHA256 mismatch: {case.case_id}")


def prepare_inputs(calibration: Path, manifest: Path, quality_path: Path, output_dir: Path,
                   expected_sha256: str = FROZEN_CALIBRATION_SHA256,
                   protocol_path: Path | None = None,
                   expected_protocol_sha256: str | None = None) -> tuple[Path, Path, dict[str, Any]]:
    """Create frozen prediction/target artifacts for admitted external rows."""
    fit = _json(calibration)
    calibration_sha = _check_frozen_inputs(calibration, fit, expected_sha256)
    if protocol_path is None or expected_protocol_sha256 is None:
        raise ValueError("external validation requires --protocol and --expected-protocol-sha256")
    protocol_sha = sha256_file(protocol_path)
    if expected_protocol_sha256 != FROZEN_PROTOCOL_SHA256 or protocol_sha != expected_protocol_sha256:
        raise ValueError("expanded validation protocol SHA is not pinned")
    cases = _external_cases(manifest)
    quality = _quality_admission(quality_path, manifest, cases)
    protocol = _json(protocol_path)
    _check_external(cases, manifest, fit, protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "nr33-external-predictions.json"
    target_path = output_dir / "nr33-external-targets.json"
    report_path = output_dir / "nr33-external-validation.json"
    plot_path = output_dir / "nr33-external-comparison.png"
    for path in (prediction_path, target_path, report_path, plot_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite external validation output: {path}")

    controls = FFTControls(**fit["controls"])
    parameters = fit["parameter_fit"]
    predictions, targets = [], []
    for case in cases:
        time, h33, metadata = load_case(case)
        qnm = independent_qnm33(metadata)
        f, nr, fft = frequency_amplitude(
            time, h33, controls, metadata["relaxation_time"], metadata["peak_time"], return_metadata=True)
        band = f >= .95 * qnm["f_ring_33"]
        f, nr = f[band], nr[band]
        eta, delta = case.q / (1 + case.q) ** 2, (case.q - 1) / (case.q + 1)
        x = 4 * eta - 1
        log_a = np.polynomial.polynomial.polyval(x, parameters["log_abs_a_lambda_over_delta_coefficients"])
        lam = np.polynomial.polynomial.polyval(x, parameters["lambda_coefficients"])
        a022 = np.pi * np.sqrt(2 * eta / 3) * (np.pi * f) ** (-7 / 6)
        predicted = a022 * delta * np.exp(log_a) * ringdown_shape(
            f, qnm["f_ring_33"], qnm["f_damp_33"], lam, fit["sigma_33"])
        common = {
            "case_id": case.case_id, "split": "holdout", "q": case.q,
            "frequency_Mf": f.tolist(), "amplitude_units": "geometric",
            "m1_msun": 50 * case.q / (1 + case.q), "m2_msun": 50 / (1 + case.q),
            "chi1": 0., "chi2": 0., "distance_mpc": 100.,
            "spin_approximation": "external NR supplied as near-nonspinning; treated as zero spin",
            "qnm": qnm, "fft": fft, "input_npz_sha256": sha256_file(Path(case.path)),
        }
        predictions.append({**common, "prediction_amplitude_geometric": predicted.tolist()})
        targets.append({**common, "target_amplitude_geometric": nr.tolist()})
    if sha256_file(calibration) != calibration_sha:
        raise ValueError("calibration changed while preparing external holdouts")
    shared = {
        "frozen": True, "split": {"rule": "original frozen training IDs plus external holdouts",
                                   "train_case_ids": list(fit["split"]["train_case_ids"]),
                                   "holdout_case_ids": [case.case_id for case in cases], "excluded_case_ids": []},
        "calibration_before_sha256": calibration_sha, "calibration_after_sha256": calibration_sha,
        "manifest_sha256": sha256_file(manifest), "adapter_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": protocol_sha,
        "quality_report_sha256": sha256_file(quality_path),
        "screened_input_manifest_sha256": quality["input_manifest_sha256"],
        "source_fit_manifest_sha256": fit["provenance"]["manifest_sha256"],
    }
    prediction_path.write_text(json.dumps({**shared, "predictions": predictions}, indent=2, allow_nan=False) + "\n")
    target_path.write_text(json.dumps({**shared, "targets": targets}, indent=2, allow_nan=False) + "\n")
    return prediction_path, target_path, {"quality": quality, "protocol_sha256": protocol_sha,
                                          "calibration_sha256": calibration_sha}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=FROZEN_CALIBRATION_SHA256)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--with-lal", action="store_true")
    args = parser.parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    prediction_path, target_path, provenance = prepare_inputs(
        args.calibration, args.manifest, args.quality, args.output_dir,
        args.expected_sha256, args.protocol, args.expected_protocol_sha256)
    report = validate_frozen_predictions(
        prediction_path, target_path, include_lal=args.with_lal,
        calibration_path=args.calibration, expected_calibration_sha256=args.expected_sha256,
        plot_path=args.output_dir / "nr33-external-comparison.png")
    report.update({
        "external": True, "external_manifest_sha256": sha256_file(args.manifest),
        "quality_report_sha256": sha256_file(args.quality),
        "quality_qualified_manifest_sha256": provenance["quality"].get("qualified_manifest", {}).get("sha256"),
        "protocol_sha256": provenance["protocol_sha256"],
        "source_fit_manifest_sha256": _json(args.calibration)["provenance"]["manifest_sha256"],
        "run": {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(), "cwd": str(Path.cwd()), "python": sys.executable,
                "command": sys.argv, "adapter_sha256": sha256_file(Path(__file__)),
                "predictions_sha256": sha256_file(prediction_path), "targets_sha256": sha256_file(target_path)},
    })
    (args.output_dir / "nr33-external-validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
