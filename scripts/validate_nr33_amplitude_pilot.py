#!/usr/bin/env python3
"""Post-freeze validation for a frozen nonspinning XHM ``(3,3)`` fit.

The validator consumes two independent, read-only artifacts:

* a frozen fit artifact containing predictions on a geometric ``Mf`` grid;
* held-out NR target amplitudes on that same grid.

An optional LALSimulation call is a third validation target.  It is made only
after the fit artifact's freeze evidence has been checked and its values never
enter a fitting call.  The JSON format is intentionally small.  A fit artifact
may put rows in ``predictions`` (or ``cases``), and a target artifact may put
rows in ``targets`` (or ``cases``).  Each row contains ``case_id``, ``m1_msun``,
``m2_msun``, ``chi1``, ``chi2``, ``distance_mpc``, ``frequency_Mf`` and an
amplitude field.  Amplitudes are geometric by default; set ``amplitude_units``
to ``physical`` when an artifact stores physical Fourier strain.

This module does not import a fitter or any fit constants.  It reports
amplitude metrics only; no complex-waveform mismatch is inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC


MODE = (3, 3)
MODE_LABEL = "33"
TINY = np.finfo(np.float64).tiny


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path*."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frequency_grid(frequency_m: Sequence[float]) -> np.ndarray:
    """Validate and return an exact, increasing geometric ``Mf`` grid.

    Equality of grids is checked with :func:`numpy.array_equal` by the caller;
    this function deliberately does not resample or round frequencies.
    """

    grid = np.asarray(frequency_m, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("frequency_Mf must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(grid)):
        raise ValueError("frequency_Mf contains a non-finite value")
    if np.any(grid <= 0.0):
        raise ValueError("frequency_Mf must be strictly positive")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("frequency_Mf must be strictly increasing")
    return grid


def total_mass_seconds(m1_msun: float, m2_msun: float) -> float:
    """Convert component masses in solar masses to total geometric seconds."""

    m1 = float(m1_msun)
    m2 = float(m2_msun)
    if not np.isfinite(m1) or not np.isfinite(m2) or m1 <= 0.0 or m2 <= 0.0:
        raise ValueError("component masses must be finite and positive")
    return (m1 + m2) * MSUN_SEC


def geometric_to_physical(
    amplitude: Sequence[float] | np.ndarray,
    total_mass_seconds_value: float,
    distance_m: float,
) -> np.ndarray:
    """Apply the Fourier-domain scale ``Msec * (Msec*c/D)``.

    The input is a dimensionless geometric Fourier mode and the output has
    physical Fourier strain units (seconds).  The factor is kept explicit so
    distance and mass normalization can be audited independently.
    """

    mass = float(total_mass_seconds_value)
    distance = float(distance_m)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("total_mass_seconds must be finite and positive")
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_m must be finite and positive")
    values = np.asarray(amplitude, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("amplitude contains a non-finite value")
    return values * (mass * (mass * C_SI / distance))


def physical_to_geometric(
    amplitude: Sequence[float] | np.ndarray,
    total_mass_seconds_value: float,
    distance_m: float,
) -> np.ndarray:
    """Undo :func:`geometric_to_physical` for physical amplitude artifacts."""

    scale = geometric_to_physical(np.asarray([1.0]), total_mass_seconds_value, distance_m)[0]
    values = np.asarray(amplitude, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("amplitude contains a non-finite value")
    return values / scale


def amplitude_metrics(predicted: Sequence[float], target: Sequence[float]) -> dict[str, Any]:
    """Return finite amplitude-only L2, norm, and maximum-error metrics."""

    prediction = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(target, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.ndim != 1 or prediction.size == 0:
        raise ValueError("predicted and target amplitudes must be a non-empty matching 1-D array")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(reference)):
        raise ValueError("predicted and target amplitudes must be finite")
    if np.any(prediction < 0.0) or np.any(reference < 0.0):
        raise ValueError("amplitudes must be non-negative")

    difference = np.abs(prediction - reference)
    target_norm = float(np.linalg.norm(reference))
    prediction_norm = float(np.linalg.norm(prediction))
    denominator = max(target_norm, TINY)
    relative = difference / np.maximum(np.abs(reference), TINY)
    return {
        "relative_amp_l2": float(np.linalg.norm(prediction - reference) / denominator),
        "norm": {
            "prediction": prediction_norm,
            "target": target_norm,
            "ratio_prediction_over_target": float(prediction_norm / denominator),
        },
        "maxerrors": {
            "absolute": float(np.max(difference)),
            "relative": float(np.max(relative)),
            "index_absolute": int(np.argmax(difference)),
            "index_relative": int(np.argmax(relative)),
        },
    }


def _lal_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def lal_one_mode_amplitude(
    frequency_m: Sequence[float],
    *,
    m1_msun: float,
    m2_msun: float,
    chi1: float = 0.0,
    chi2: float = 0.0,
    distance_m: float,
    lal_module: Any | None = None,
    lalsimulation_module: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate LAL's public one-mode API on the exact ``Mf`` grid.

    Positive-frequency ``(3,3)`` uses the public API's ``m=-3`` argument.
    Returned values are physical Fourier strain; only their magnitudes are
    returned.  Imports are local so the core NR validation remains usable
    without LALSuite.
    """

    grid = validate_frequency_grid(frequency_m)
    if lal_module is None or lalsimulation_module is None:
        import lal as lal_module  # type: ignore[no-redef]
        import lalsimulation as lalsimulation_module  # type: ignore[no-redef]

    mass = total_mass_seconds(m1_msun, m2_msun)
    frequency_hz = grid / mass  # exact conversion from the supplied Mf values
    vector = lal_module.CreateREAL8Vector(int(grid.size))
    vector.data[:] = frequency_hz
    params = lal_module.CreateDict()
    release_version = "unavailable"
    release_source = "public lookup unavailable"
    setter = getattr(lalsimulation_module, "SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion", None)
    lookup = getattr(lalsimulation_module, "SimInspiralWaveformParamsLookupPhenomXHMReleaseVersion", None)
    if setter is not None and lookup is not None:
        try:
            setter(params, 122022)
            release_version = int(lookup(params))
            release_source = "explicit public setter (requested 122022)"
        except Exception:  # pragma: no cover - depends on LALSuite bindings
            release_version = "unavailable"
            release_source = "public setter failed"
    series = lalsimulation_module.SimIMRPhenomXHMFrequencySequenceOneMode(
        vector,
        float(m1_msun) * lal_module.MSUN_SI,
        float(m2_msun) * lal_module.MSUN_SI,
        float(chi1),
        float(chi2),
        MODE[0],
        -MODE[1],
        float(distance_m),
        0.0,
        20.0,
        params,
    )
    values = np.asarray(series.data.data, dtype=np.complex128)
    if values.shape != grid.shape:
        raise ValueError("LAL returned a series with a different shape")
    amplitudes = np.abs(values).astype(np.float64, copy=True)
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("LAL returned a non-finite amplitude")
    return amplitudes, {
        "api": "SimIMRPhenomXHMFrequencySequenceOneMode",
        "mode_argument": [3, -3],
        "frequency_hz": frequency_hz.tolist(),
        "units": "physical Fourier-domain strain seconds",
        "lal_version": _lal_version(lal_module),
        "lalsimulation_version": _lal_version(lalsimulation_module),
        "xhm_release_version": release_version,
        "xhm_release_version_source": release_source,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("metadata")
    if isinstance(nested, Mapping):
        merged = dict(payload)
        merged.update(nested)
        return merged
    nested = payload.get("fit_artifact")
    if isinstance(nested, Mapping):
        merged = dict(payload)
        merged.update(nested)
        return merged
    return payload


def _hash_field(metadata: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = metadata.get(name)
        if value is not None and str(value):
            return str(value)
    hashes = metadata.get("hashes")
    if isinstance(hashes, Mapping):
        for name in names:
            value = hashes.get(name)
            if value is not None and str(value):
                return str(value)
    return None


def _digest(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    return digest


def _read_digest_sidecar(path: str | Path) -> str:
    """Read a digest sidecar containing a bare SHA-256 or a small JSON object."""

    raw = Path(path).read_text().strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    if isinstance(parsed, Mapping):
        for key in ("sha256", "digest", "calibration_sha256", "calibration_digest"):
            if key in parsed:
                return _digest(parsed[key], f"digest sidecar {key}")
        raise ValueError("digest sidecar JSON has no sha256/digest field")
    match = re.fullmatch(r"[0-9a-fA-F]{64}(?:\s+.*)?", str(parsed).strip())
    if not match:
        raise ValueError("digest sidecar must contain a SHA-256 digest")
    return _digest(str(parsed).split()[0], "digest sidecar")


def require_frozen_artifact(
    path: str | Path,
    payload: Mapping[str, Any] | None = None,
    *,
    calibration_path: str | Path | None = None,
    expected_calibration_sha256: str | None = None,
    calibration_digest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Check freeze state and hash the actual calibration and artifact bytes.

    The calibration digest comes from the file and an independently supplied
    expected digest (usually a sidecar).  A digest embedded in the JSON fit
    artifact is recorded as a claim only: embedding a file's own SHA in that
    file cannot provide a self-consistent integrity check.
    """

    artifact_path = Path(path)
    value = dict(payload) if payload is not None else _read_json(artifact_path)
    metadata = _metadata(value)
    status = str(metadata.get("status", "")).lower()
    if metadata.get("frozen") is not True and status not in {"frozen", "complete", "completed"}:
        raise ValueError("fit artifact is not frozen; validation requires status=frozen or frozen=true")

    declared_before = _hash_field(metadata, "calibration_before_sha256", "calibration_before_hash")
    declared_after = _hash_field(metadata, "calibration_after_sha256", "calibration_after_hash")
    if calibration_path is None:
        if not declared_before or not declared_after:
            raise ValueError("frozen artifact must record calibration before and after hashes")
        before = _digest(declared_before, "calibration before hash")
        after = _digest(declared_after, "calibration after hash")
        if before != after:
            raise ValueError("calibration source changed between before and after hashes")
        calibration_actual = False
        expected = before
    else:
        calibration_file = Path(calibration_path)
        if not calibration_file.is_file():
            raise ValueError(f"calibration file does not exist: {calibration_file}")
        if expected_calibration_sha256 and calibration_digest_path:
            raise ValueError("provide one of expected_calibration_sha256 or calibration_digest_path")
        expected = (_digest(expected_calibration_sha256, "expected calibration hash")
                    if expected_calibration_sha256 else
                    _read_digest_sidecar(calibration_digest_path) if calibration_digest_path else None)
        if expected is None:
            raise ValueError("disk validation requires an independent expected calibration digest")
        before = sha256_file(calibration_file)
        if before != expected:
            raise ValueError("calibration file does not match the expected frozen digest")
        if declared_before and _digest(declared_before, "declared calibration before hash") != before:
            raise ValueError("artifact calibration before hash disagrees with the calibration file")
        if declared_after and _digest(declared_after, "declared calibration after hash") != before:
            raise ValueError("artifact calibration after hash disagrees with the calibration file")
        calibration_actual = True
    after = before

    artifact_before = sha256_file(artifact_path)
    declared_artifact = _hash_field(metadata, "fit_artifact_sha256", "artifact_sha256")
    return {
        "status": status or "frozen",
        "calibration_before_sha256": before,
        "calibration_after_sha256": after,
        "calibration_path": str(calibration_path) if calibration_path is not None else None,
        "calibration_digest_independently_checked": calibration_actual,
        "calibration_expected_sha256": expected,
        "fit_artifact_sha256_before": artifact_before,
        "fit_artifact_sha256": artifact_before,
        "fit_artifact_sha256_declared": declared_artifact,
        "_calibration_path": calibration_path,
        "_artifact_path": artifact_path,
    }


def _finish_hash_guard(guard: dict[str, Any]) -> dict[str, Any]:
    """Hash sources again after validation and reject any mutation."""

    calibration_path = guard.get("_calibration_path")
    artifact_path = guard.get("_artifact_path")
    if calibration_path is not None:
        after = sha256_file(calibration_path)
        if after != guard["calibration_before_sha256"]:
            raise ValueError("calibration file changed during validation")
    else:
        after = guard["calibration_before_sha256"]
    artifact_after = sha256_file(artifact_path) if artifact_path is not None else guard.get("fit_artifact_sha256_before")
    if artifact_after != guard.get("fit_artifact_sha256_before"):
        raise ValueError("fit artifact changed during validation")
    guard["calibration_after_sha256"] = after
    guard["fit_artifact_sha256_after"] = artifact_after
    return guard


def _rows(payload: Mapping[str, Any], role: str) -> list[Mapping[str, Any]]:
    keys = ("predictions", "cases", "rows") if role == "prediction" else ("targets", "cases", "rows")
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            if not value:
                raise ValueError(f"{key} must not be empty")
            if not all(isinstance(row, Mapping) for row in value):
                raise ValueError(f"{key} must contain JSON objects")
            return list(value)
    raise ValueError(f"artifact has no {role} rows (expected one of {keys})")


def _case_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("case_id", "id", "label", "name", "key"):
        if key in row:
            value = str(row[key]).strip()
            if not value:
                raise ValueError(f"row {index} has an empty case ID")
            return value
    raise ValueError(f"row {index} is missing an explicit case ID")


def _index_rows(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        case_id = _case_id(row, index)
        if case_id in indexed:
            raise ValueError(f"duplicate {role} case ID {case_id!r}")
        indexed[case_id] = row
    return indexed


def _declared_split_ids(payload: Mapping[str, Any]) -> tuple[set[str] | None, set[str] | None]:
    """Return declared (training, heldout) IDs from common artifact layouts."""

    containers: list[Mapping[str, Any]] = [payload]
    split = payload.get("split")
    if isinstance(split, Mapping):
        containers.append(split)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(metadata)
        metadata_split = metadata.get("split")
        if isinstance(metadata_split, Mapping):
            containers.append(metadata_split)
    names = {
        "train": ("train_case_ids", "training_case_ids", "train_ids", "training_ids"),
        "holdout": ("holdout_case_ids", "heldout_case_ids", "holdout_ids", "heldout_ids"),
    }
    found: dict[str, set[str] | None] = {"train": None, "holdout": None}
    for container in containers:
        for kind, keys in names.items():
            for key in keys:
                value = container.get(key)
                if isinstance(value, (list, tuple, set)):
                    raw_ids = [str(item).strip() for item in value]
                    if len(raw_ids) != len(set(raw_ids)):
                        raise ValueError(f"declared {kind} IDs must be unique")
                    ids = set(raw_ids)
                    if not ids or "" in ids:
                        raise ValueError(f"declared {kind} IDs must be non-empty")
                    if found[kind] is not None and found[kind] != ids:
                        raise ValueError(f"conflicting declared {kind} ID sets")
                    found[kind] = ids
                    break
    return found["train"], found["holdout"]


def _grid_from(row: Mapping[str, Any]) -> np.ndarray:
    for key in ("frequency_Mf", "frequency_m", "geometric_frequency", "frequency_geom", "Mf"):
        if key in row:
            return validate_frequency_grid(row[key])
    raise ValueError("row is missing frequency_Mf")


def _amplitude_from(row: Mapping[str, Any], role: str) -> tuple[np.ndarray, str]:
    names = (
        ("prediction_amplitude_geometric", "predicted_amplitude_geometric", "prediction_amplitude_physical", "predicted_amplitude_physical", "amplitude_geometric", "prediction_amplitude", "predicted_amplitude", "prediction_amp", "amplitude_physical", "amplitudes", "amplitude")
        if role == "prediction"
        else ("target_amplitude_geometric", "target_amplitude_physical", "amplitude_geometric", "target_amplitude", "target_amp", "amplitude_physical", "amplitudes", "amplitude")
    )
    for key in names:
        if key in row:
            units = str(row.get("amplitude_units", "geometric")).lower()
            if "physical" in key:
                units = "physical"
            if units not in {"geometric", "physical"}:
                raise ValueError(f"unsupported amplitude_units={units!r}")
            values = np.asarray(row[key], dtype=np.float64)
            if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{key} must be a finite non-negative 1-D array")
            return values, units
    raise ValueError(f"row is missing a {role} amplitude field")


def _parameters(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    aliases = {
        "m1_msun": ("m1_msun", "m1"),
        "m2_msun": ("m2_msun", "m2"),
        "chi1": ("chi1", "chi1z", "spin1z"),
        "chi2": ("chi2", "chi2z", "spin2z"),
        "distance_mpc": ("distance_mpc", "distance"),
    }
    values: dict[str, Any] = {}
    for name, options in aliases.items():
        found = next((key for key in options if key in row), None)
        if found is None:
            raise ValueError(f"row is missing one of {options}")
        values[name] = row[found]
    m1, m2, chi1, chi2, distance = (float(values[key]) for key in aliases)
    total_mass_seconds(m1, m2)
    if not all(np.isfinite(value) for value in (chi1, chi2, distance)) or distance <= 0.0:
        raise ValueError("spins and distance must be finite, with positive distance")
    if abs(chi1) > 1.0 or abs(chi2) > 1.0:
        raise ValueError("dimensionless spins must be in [-1, 1]")
    return m1, m2, chi1, chi2, distance


def validate_frozen_predictions(
    fit_artifact: str | Path | Mapping[str, Any],
    target_artifact: str | Path | Mapping[str, Any],
    *,
    include_lal: bool = False,
    plot_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    expected_calibration_sha256: str | None = None,
    calibration_digest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate frozen predictions against held-out targets and optional LAL."""

    fit_path = fit_artifact if isinstance(fit_artifact, (str, Path)) else None
    target_path = target_artifact if isinstance(target_artifact, (str, Path)) else None
    fit = _read_json(fit_artifact) if fit_path is not None else dict(fit_artifact)
    target = _read_json(target_artifact) if target_path is not None else dict(target_artifact)
    if fit_path is not None:
        freeze = require_frozen_artifact(
            fit_path, fit, calibration_path=calibration_path,
            expected_calibration_sha256=expected_calibration_sha256,
            calibration_digest_path=calibration_digest_path,
        )
    else:
        freeze = _require_frozen_mapping(fit)

    prediction_rows = _rows(fit, "prediction")
    target_rows = _rows(target, "target")
    prediction_by_id = _index_rows(prediction_rows, "prediction")
    target_by_id = _index_rows(target_rows, "target")
    missing = sorted(set(prediction_by_id) - set(target_by_id))
    unexpected = sorted(set(target_by_id) - set(prediction_by_id))
    if missing or unexpected:
        raise ValueError(f"held-out ID mismatch: missing_targets={missing}, unexpected_targets={unexpected}")
    fit_train, fit_holdout = _declared_split_ids(fit)
    target_train, target_holdout = _declared_split_ids(target)
    if fit_train is None or fit_holdout is None:
        raise ValueError("frozen fit artifact must declare training and held-out case IDs")
    if fit_train & fit_holdout:
        raise ValueError("training and held-out case IDs overlap")
    if set(prediction_by_id) != fit_holdout:
        raise ValueError("prediction IDs do not exactly match the frozen held-out ID declaration")
    if target_holdout is not None and target_holdout != fit_holdout:
        raise ValueError("target held-out IDs disagree with the frozen fit split")
    if target_train is not None and target_train != fit_train:
        raise ValueError("target training IDs disagree with the frozen fit split")
    if target_train is not None and target_train & fit_holdout:
        raise ValueError("target training IDs overlap frozen held-out IDs")
    results: list[dict[str, Any]] = []
    lal_rows: list[dict[str, Any]] = []
    plot_curves: list[dict[str, np.ndarray | str]] = []
    lal_pending: list[tuple[dict[str, Any], np.ndarray, np.ndarray, float, tuple[float, float, float, float, float], np.ndarray]] = []

    for case_id, prediction_row in prediction_by_id.items():
        target_row = target_by_id[case_id]
        grid = _grid_from(prediction_row)
        target_grid = _grid_from(target_row)
        if not np.array_equal(grid, target_grid):
            raise ValueError(f"case {case_id!r} prediction and target grids differ exactly")
        params = _parameters(prediction_row)
        target_params = _parameters(target_row)
        if not np.array_equal(np.asarray(params, dtype=np.float64), np.asarray(target_params, dtype=np.float64)):
            raise ValueError(f"case {case_id!r} prediction and target parameters differ")
        mass = total_mass_seconds(params[0], params[1])
        distance_m = params[4] * MPC_SI
        prediction, prediction_units = _amplitude_from(prediction_row, "prediction")
        target_values, target_units = _amplitude_from(target_row, "target")
        if prediction.shape != grid.shape or target_values.shape != grid.shape:
            raise ValueError(f"case {case_id!r} amplitude shape does not match frequency_Mf")
        prediction_geom = prediction if prediction_units == "geometric" else physical_to_geometric(prediction, mass, distance_m)
        target_geom = target_values if target_units == "geometric" else physical_to_geometric(target_values, mass, distance_m)
        metric = amplitude_metrics(prediction_geom, target_geom)
        physical_scale = mass * (mass * C_SI / distance_m)
        target_physical = geometric_to_physical(target_geom, mass, distance_m)
        prediction_physical = geometric_to_physical(prediction_geom, mass, distance_m)
        metric["relativeampL2"] = metric["relative_amp_l2"]
        metric["norm_physical"] = {
            "prediction": float(np.linalg.norm(prediction_physical)),
            "target": float(np.linalg.norm(target_physical)),
            "ratio_prediction_over_target": float(np.linalg.norm(prediction_physical) / max(np.linalg.norm(target_physical), TINY)),
        }
        target_split = target_row.get("split")
        if target_split is None or not str(target_split).strip():
            raise ValueError(f"case {case_id!r} target is missing an explicit held-out split")
        prediction_split = prediction_row.get("split")
        if prediction_split is not None and str(prediction_split).lower() not in {"heldout", "holdout", "test"}:
            raise ValueError(f"case {case_id!r} prediction is not marked as held-out")
        metric.update(
            {
                "case_id": case_id,
                "mode": list(MODE),
                "frequency_Mf": grid.tolist(),
                "m1_msun": params[0],
                "m2_msun": params[1],
                "chi1": params[2],
                "chi2": params[3],
                "distance_mpc": params[4],
                "amplitude_comparison_units": "geometric",
                "physical_scale": physical_scale,
                "normalization": {
                    "formula": "Msec*(Msec*c/D)",
                    "total_mass_seconds": mass,
                    "distance_m": distance_m,
                    "roundtrip_max_abs_error": float(np.max(np.abs(physical_to_geometric(target_physical, mass, distance_m) - target_geom))),
                },
                "target_split": target_split,
            }
        )
        if str(metric["target_split"]).lower() not in {"heldout", "holdout", "test"}:
            raise ValueError(f"case {case_id!r} is not marked as held-out")
        lal_geom = None
        if include_lal:
            lal_pending.append((metric, prediction_physical, target_physical, mass, params, grid))
        plot_curves.append(
            {
                "case_id": case_id,
                "frequency_Mf": grid.copy(),
                "nr": target_geom.copy(),
                "frozen_fit": prediction_geom.copy(),
                "lal": lal_geom,
            }
        )
        results.append(metric)

    # This second hash is deliberately taken before the optional external
    # target is called.  Thus LAL can never run against a changed calibration.
    freeze = _finish_hash_guard(freeze)
    if include_lal:
        for metric, prediction_physical, target_physical, mass, params, grid in lal_pending:
            lal_amplitude, lal_meta = lal_one_mode_amplitude(
                grid, m1_msun=params[0], m2_msun=params[1], chi1=params[2],
                chi2=params[3], distance_m=params[4] * MPC_SI,
            )
            metric["lal_validation"] = {
                "prediction_vs_lal": amplitude_metrics(prediction_physical, lal_amplitude),
                "target_vs_lal": amplitude_metrics(target_physical, lal_amplitude),
                **lal_meta,
            }
            lal_rows.append({"case_id": metric["case_id"], **lal_meta})
            lal_geom = physical_to_geometric(lal_amplitude, mass, params[4] * MPC_SI)
            next(curve for curve in plot_curves if curve["case_id"] == metric["case_id"])["lal"] = lal_geom

    plot_metadata: dict[str, Any] = {"requested": plot_path is not None, "status": "not_requested"}
    if plot_path is not None:
        plot_metadata = _write_plot(plot_path, plot_curves)
    # Recheck after the optional LAL call and plot generation as well.  The
    # reported after hash therefore covers the complete validation interval.
    freeze = _finish_hash_guard(freeze)
    freeze.pop("_calibration_path", None)
    freeze.pop("_artifact_path", None)

    return {
        "status": "complete",
        "purpose": "post-freeze held-out NR amplitude validation for XHM mode (3,3)",
        "mode": list(MODE),
        "amplitude_only": True,
        "complex_waveform_mismatch_evaluated": False,
        "freeze": freeze,
        "calibration_before_sha256": freeze["calibration_before_sha256"],
        "calibration_after_sha256": freeze["calibration_after_sha256"],
        "frequency_grid_exact": True,
        "cases": results,
        "lal_validation": {"enabled": include_lal, "cases": lal_rows},
        "plot": plot_metadata,
        "versions": _versions(include_lal),
        "source_sha256": {"validator": sha256_file(Path(__file__))},
    }


def _write_plot(path: str | Path, curves: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write a compact review plot, when matplotlib is installed.

    All curves are plotted in the common geometric amplitude convention.  The
    lower panel contains fractional amplitude residuals; it makes no claim
    about complex phase or waveform mismatch.
    """

    destination = Path(path)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"requested": True, "status": "skipped_matplotlib_unavailable", "path": str(destination)}
    if not curves:
        raise ValueError("cannot plot an empty validation result")
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.5), sharex=True, constrained_layout=True)
    for curve in curves:
        x = np.asarray(curve["frequency_Mf"], dtype=np.float64)
        nr = np.asarray(curve["nr"], dtype=np.float64)
        fit = np.asarray(curve["frozen_fit"], dtype=np.float64)
        label = str(curve["case_id"])
        axes[0].plot(x, nr, "o-", ms=2.5, lw=1.0, label=f"NR {label}")
        axes[0].plot(x, fit, "--", lw=1.0, label=f"frozen fit {label}")
        lal_values = curve.get("lal")
        if lal_values is not None:
            axes[0].plot(x, np.asarray(lal_values, dtype=np.float64), ":", lw=1.0, label=f"LAL {label}")
        safe = np.maximum(np.abs(nr), TINY)
        axes[1].plot(x, (fit - nr) / safe, "--", lw=1.0, label=f"fit−NR {label}")
        if lal_values is not None:
            axes[1].plot(x, (np.asarray(lal_values, dtype=np.float64) - nr) / safe, ":", lw=1.0, label=f"LAL−NR {label}")
    axes[0].set_ylabel("geometric |H₃₃|")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel("Mf")
    axes[1].set_ylabel("fractional amplitude residual")
    axes[1].axhline(0.0, color="black", lw=0.6)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return {"requested": True, "status": "written", "path": str(destination), "sha256": sha256_file(destination)}


def _require_frozen_mapping(fit: Mapping[str, Any]) -> dict[str, Any]:
    """In-memory variant used by tests; disk artifacts get full SHA checks."""

    metadata = _metadata(fit)
    status = str(metadata.get("status", "")).lower()
    if metadata.get("frozen") is not True and status not in {"frozen", "complete", "completed"}:
        raise ValueError("fit artifact is not frozen")
    before = _hash_field(metadata, "calibration_before_sha256", "calibration_before_hash")
    after = _hash_field(metadata, "calibration_after_sha256", "calibration_after_hash")
    if not before or not after or before != after:
        raise ValueError("frozen artifact must contain matching calibration before/after hashes")
    return {
        "status": status or "frozen",
        "calibration_before_sha256": _digest(before, "calibration before hash"),
        "calibration_after_sha256": _digest(after, "calibration after hash"),
        "calibration_expected_sha256": _digest(before, "calibration expected hash"),
        "calibration_digest_independently_checked": False,
        "fit_artifact_sha256_before": None,
        "fit_artifact_sha256": None,
        "fit_artifact_sha256_declared": None,
        "_calibration_path": None,
        "_artifact_path": None,
    }


def _versions(include_lal: bool) -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    if include_lal:
        try:
            import lal
            import lalsimulation as ls
            versions.update({"lal": _lal_version(lal), "lalsimulation": _lal_version(ls)})
        except ImportError:
            versions.update({"lal": "unavailable", "lalsimulation": "unavailable"})
    return versions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-artifact", "--artifact", dest="fit_artifact", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True,
                        help="frozen coefficient file whose bytes are hashed")
    digest = parser.add_mutually_exclusive_group(required=True)
    digest.add_argument("--calibration-digest", type=Path,
                        help="sidecar containing the expected frozen calibration SHA-256")
    digest.add_argument("--calibration-sha256",
                        help="expected frozen calibration SHA-256")
    parser.add_argument("--with-lal", action="store_true", help="also compare physical amplitudes with public LAL (validation only)")
    parser.add_argument("--plot", type=Path, help="optional compact NR/frozen-fit/LAL amplitude review plot")
    args = parser.parse_args(argv)
    report = validate_frozen_predictions(
        args.fit_artifact, args.targets, include_lal=args.with_lal, plot_path=args.plot,
        calibration_path=args.calibration,
        expected_calibration_sha256=args.calibration_sha256,
        calibration_digest_path=args.calibration_digest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
