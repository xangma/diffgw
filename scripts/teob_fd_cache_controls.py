#!/usr/bin/env python3
"""Cache-only controls for the clean TEOB FD attribution experiment.

This script deliberately does not evolve an EOB trajectory.  It re-applies
the native TD conditioner to the saved raw polarizations, compares that replay
with the saved native FD result, and scores the native result against the
saved public ``SimInspiralFD`` result.  No phase or time correction is ever
applied to an output; the linear phase and epoch-factor calculations are
diagnostics only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import socket
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from torchwave.approximants.eob.td_fd_conditioning_clean import (  # noqa: E402
    TDSupportPlan,
    condition_td_to_fd,
    plan_td_support,
)

DEFAULT_ROOT = ROOT / "docs/reviews/2026-09-15-teob-fd-attribution"
POLARIZATIONS = ("plus", "cross")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array(z: np.lib.npyio.NpzFile, *names: str) -> np.ndarray | None:
    for name in names:
        if name in z.files:
            return np.asarray(z[name])
    return None


def _scalar(z: np.lib.npyio.NpzFile, *names: str) -> float | None:
    value = _array(z, *names)
    if value is None or value.size != 1:
        return None
    out = float(value.reshape(()))
    return out if math.isfinite(out) else None


def _require(z: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    value = _array(z, *names)
    if value is None:
        raise KeyError(f"cache is missing one of: {', '.join(names)}")
    return value


def _finite_float(value: Any, name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise KeyError(f"request metadata is missing {name}")
        return default
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"request metadata {name} is not finite")
    return out


def _request(report: dict[str, Any], cache: np.lib.npyio.NpzFile) -> dict[str, float]:
    case = report.get("case", {})
    params = case.get("params", {}) if isinstance(case, dict) else {}
    grid = case.get("grid", {}) if isinstance(case, dict) else {}
    return {
        "mass1": _finite_float(params.get("mass1"), "mass1"),
        "mass2": _finite_float(params.get("mass2"), "mass2"),
        "flow": _finite_float(grid.get("flow"), "flow"),
        "fhigh": _finite_float(grid.get("fhigh"), "fhigh"),
        "df": _finite_float(grid.get("df"), "df"),
        "chi1z": _finite_float(params.get("chi1", params.get("chi1z", 0.0)), "chi1z", 0.0),
        "chi2z": _finite_float(params.get("chi2", params.get("chi2z", 0.0)), "chi2z", 0.0),
        "tc": _finite_float(params.get("tc"), "tc", 0.0),
        "distance": _finite_float(params.get("distance"), "distance", 100.0),
        "inclination": _finite_float(params.get("inclination"), "inclination", 0.0),
        "phic": _finite_float(params.get("phic"), "phic", 0.0),
        "lambda1": _finite_float(params.get("lambda1"), "lambda1", 0.0),
        "lambda2": _finite_float(params.get("lambda2"), "lambda2", 0.0),
        "dt": _finite_float(_scalar(cache, "sampling_dt_seconds"), "sampling_dt_seconds"),
    }


def _condition(raw_hp: np.ndarray, raw_hc: np.ndarray, request: dict[str, float], *, support: TDSupportPlan | None = None, dt: float | None = None):
    if support is None:
        support = plan_td_support(
            request["mass1"], request["mass2"], request["flow"],
            request["chi1z"], request["chi2z"], dtype=torch.float64,
        )
    return condition_td_to_fd(
        torch.from_numpy(np.asarray(raw_hp, dtype=np.float64)),
        torch.from_numpy(np.asarray(raw_hc, dtype=np.float64)),
        dt=request["dt"] if dt is None else dt, delta_f=request["df"], f_max=request["fhigh"],
        support=support, tc=request["tc"], return_metadata=True,
    )


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / denominator) if denominator else float(np.linalg.norm(a - b))


def _fd_metrics(value: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    value = np.asarray(value)
    reference = np.asarray(reference)
    value_norm = float(np.linalg.norm(value))
    reference_norm = float(np.linalg.norm(reference))
    value_mag = np.abs(value)
    reference_mag = np.abs(reference)
    return {
        "complex_relative_l2": _relative(value, reference),
        "max_complex_abs_error": float(np.max(np.abs(value - reference))) if value.size else 0.0,
        "magnitude_relative_l2": _relative(value_mag, reference_mag),
        "norm": value_norm,
        "reference_norm": reference_norm,
        "norm_relative_error": abs(value_norm - reference_norm) / reference_norm if reference_norm else value_norm,
        "mean_magnitude": float(np.mean(value_mag)) if value.size else 0.0,
        "reference_mean_magnitude": float(np.mean(reference_mag)) if value.size else 0.0,
        "samples": int(value.size),
    }


def _shared_bins(native_f: np.ndarray, public_f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact shared bins; never interpolate or fit a frequency shift."""
    native_f = np.asarray(native_f, dtype=np.float64).reshape(-1)
    public_f = np.asarray(public_f, dtype=np.float64).reshape(-1)
    j = np.searchsorted(public_f, native_f)
    valid = j < public_f.size
    valid[valid] &= native_f[valid] == public_f[j[valid]]
    i = np.flatnonzero(valid)
    return native_f[i], i, j[i]


def _bands(flow: float, fhigh: float) -> list[tuple[float, float]]:
    edges = sorted(set([float(flow), 60.0, 100.0, 200.0, 500.0, float(fhigh)]))
    return [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo and lo < fhigh and hi > flow]


def _band_metrics(freq: np.ndarray, value: np.ndarray, reference: np.ndarray, flow: float, fhigh: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lo, hi in _bands(flow, fhigh):
        mask = (freq >= lo) & (freq < hi)
        out[f"{lo:g}-{hi:g}Hz"] = {
            "bounds_hz": [lo, hi],
            "valid_bins": int(mask.sum()),
            "metrics": _fd_metrics(value[mask], reference[mask]) if mask.any() else None,
        }
    return out


def _phase_ratio(freq: np.ndarray, value: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    mask = np.isfinite(value) & np.isfinite(reference) & (np.abs(reference) > max(scale * 1e-12, 1e-300))
    if mask.sum() < 3:
        return {"status": "insufficient_nonzero_reference", "valid_bins": int(mask.sum()), "applied": False}
    ratio = value[mask] / reference[mask]
    phase = np.unwrap(np.angle(ratio))
    design = np.column_stack((freq[mask], np.ones(int(mask.sum()))))
    slope, intercept = np.linalg.lstsq(design, phase, rcond=None)[0]
    residual = phase - (slope * freq[mask] + intercept)
    return {
        "status": "diagnostic_only",
        "applied": False,
        "valid_bins": int(mask.sum()),
        "slope_rad_per_hz": float(slope),
        "intercept_rad": float(intercept),
        "equivalent_time_shift_seconds": float(-slope / (2.0 * np.pi)),
        "phase_fit_rms_rad": float(np.sqrt(np.mean(residual * residual))),
        "phase_fit_max_abs_rad": float(np.max(np.abs(residual))),
        "ratio_magnitude_median": float(np.median(np.abs(ratio))),
        "ratio_magnitude_mean": float(np.mean(np.abs(ratio))),
    }


def _td_control(native: np.lib.npyio.NpzFile, public: np.lib.npyio.NpzFile | None, *, attachment_time: float | None = None, cached_attachment_index: int | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if public is None:
        return {"status": "not_run", "reason": "no --public-td-npz supplied; no evolution was started"}, {}
    ntime = _array(native, "td_time", "time")
    ptime = _array(public, "td_time", "time", "exact_start_time")
    if ntime is None:
        ntime = np.arange(_require(native, "h_plus_td").size, dtype=np.float64)
    if ptime is None:
        ptime = np.arange(_require(public, "exact_start_raw_h_plus", "h_plus_td").size, dtype=np.float64)
    nhp = _require(native, "h_plus_td")
    nhc = _require(native, "h_cross_td")
    php = _require(public, "exact_start_raw_h_plus", "h_plus_td")
    phc = _require(public, "exact_start_raw_h_cross", "h_cross_td")
    n = min(nhp.size, php.size, ntime.size, ptime.size)
    data: dict[str, np.ndarray] = {"td_time_shared": np.asarray(ntime[:n]), "native_hp_shared": nhp[:n], "public_hp_shared": php[:n], "native_hc_shared": nhc[:n], "public_hc_shared": phc[:n]}
    result: dict[str, Any] = {"status": "complete", "comparison": "same elapsed array indices; no epoch alignment or fitted shift", "shared_samples": int(n), "native_samples": int(nhp.size), "public_samples": int(php.size), "native_unmatched_tail_samples": max(0, int(nhp.size - n)), "public_unmatched_tail_samples": max(0, int(php.size - n)), "polarizations": {}}
    # event_attachment_index belongs to the joined auxiliary assembly grid, not
    # the uniform TD cache.  Split the actual shared elapsed-index comparison
    # by the physical event time on the cached TD time axis.
    split = None
    if attachment_time is not None:
        split = int(np.searchsorted(np.asarray(ntime, dtype=np.float64), float(attachment_time), side="left"))
        split = min(max(split, 0), n)
    if split is not None:
        result["attachment_split"] = {"event_attachment_time": float(attachment_time), "cached_joined_attachment_index": int(cached_attachment_index) if cached_attachment_index is not None else None, "uniform_td_attachment_index": int(split), "shared_split_index": int(split), "split_rule": "searchsorted(native td_time, event_attachment_time, side='left')", "pre_definition": "shared elapsed indices [0, split)", "post_definition": "shared elapsed indices [split, shared_samples)"}
    for pol, na, pa in (("plus", nhp, php), ("cross", nhc, phc)):
        common_n = na[:n]
        common_p = pa[:n]
        ntail = na[n:]
        ptail = pa[n:]
        nden = float(np.dot(na, na)) or 1.0
        pden = float(np.dot(pa, pa)) or 1.0
        row = {"shared_elapsed_index_relative_l2": _relative(common_n, common_p), "native_unmatched_tail_energy_fraction": float(np.dot(ntail, ntail) / nden), "public_unmatched_tail_energy_fraction": float(np.dot(ptail, ptail) / pden)}
        if split is not None:
            row["pre_attachment_relative_l2"] = _relative(common_n[:split], common_p[:split])
            row["post_attachment_relative_l2"] = _relative(common_n[split:n], common_p[split:n])
        result["polarizations"][pol] = row
    return result, data


def _validate_public_td_sidecar(sidecar: dict[str, Any], request: dict[str, float]) -> tuple[float, TDSupportPlan]:
    """Require a sidecar that proves the optional TD fixture is the same case."""
    api = sidecar.get("api_parameters", {})
    requested = sidecar.get("request", {})
    if not isinstance(api, dict) or not isinstance(requested, dict):
        raise ValueError("public TD sidecar lacks request/API metadata")
    api_checks = {
        "m1_msun": request["mass1"], "m2_msun": request["mass2"],
        "requested_f_low_hz": request["flow"], "f_max_hz": request["fhigh"],
        "delta_f_hz": request["df"], "inclination_rad": request["inclination"],
        "phi_ref_rad": request["phic"],
    }
    request_checks = {
        "mass1_msun": request["mass1"], "mass2_msun": request["mass2"],
        "requested_f_low_hz": request["flow"], "f_max_hz": request["fhigh"],
        "delta_f_hz": request["df"], "inclination_rad": request["inclination"],
        "phi_ref_rad": request["phic"],
    }
    for label, metadata, checks in (("api_parameters", api, api_checks), ("request", requested, request_checks)):
        for key, expected in checks.items():
            # Request metadata is decimal JSON generated from the same request;
            # retain a small tolerance only for harmless JSON float spelling.
            if key not in metadata or not math.isclose(float(metadata[key]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"public TD sidecar {label} does not match request: {key}={metadata.get(key)!r}, expected {expected}")
    for key, expected in (("distance_mpc", request["distance"]), ("delta_t_seconds", request["dt"])):
        if key not in api or float(api[key]) != expected:
            raise ValueError(f"public TD sidecar api_parameters does not match request: {key}={api.get(key)!r}, expected {expected}")
    for key, expected in (("distance_mpc", request["distance"]), ("delta_t_seconds", request["dt"]), ("tc_seconds", request["tc"])):
        if key not in requested or float(requested[key]) != expected:
            raise ValueError(f"public TD sidecar request does not match request: {key}={requested.get(key)!r}, expected {expected}")
    if requested.get("api") not in (None, "SimIMRTEOBResumS"):
        raise ValueError(f"public TD sidecar request uses unexpected API {requested.get('api')!r}")
    if api.get("mode") not in (None, [2, 2]):
        raise ValueError(f"public TD sidecar uses unexpected mode {api.get('mode')!r}")
    if float(api.get("PN_tidal_order", 0.0)) != 0.0:
        raise ValueError("public TD sidecar uses nonzero PN tidal order")
    opts = sidecar.get("options", {})
    if not isinstance(opts, dict) or float(opts.get("lambda1", 0.0)) != request["lambda1"] or float(opts.get("lambda2", 0.0)) != request["lambda2"]:
        raise ValueError("public TD sidecar tidal options do not match request")
    spins = requested.get("spins", [0.0] * 6)
    # The clean cache records only aligned z spins.  The public fixture must
    # have zero transverse spins and matching z entries for both bodies.
    expected_spins = [0.0, 0.0, request["chi1z"], 0.0, 0.0, request["chi2z"]]
    if len(spins) != 6 or any(abs(float(x) - y) > 1e-12 for x, y in zip(spins, expected_spins)):
        raise ValueError(f"public TD sidecar spins do not match request: {spins!r}")
    sign = str(sidecar.get("scope", {}).get("direct_to_generic_sign", ""))
    if "multiplied by -1" not in sign:
        raise ValueError("public TD sidecar lacks the reviewed direct-to-generic sign contract")
    sampling = sidecar.get("sampling", {})
    dt = float(sampling.get("dt_seconds", api.get("delta_t_seconds", 0.0)))
    if not math.isfinite(dt) or dt <= 0 or dt != request["dt"]:
        raise ValueError("public TD sidecar lacks a positive sampling dt")
    clean = sidecar.get("public_support_plan", sidecar.get("clean_plan"))
    if not isinstance(clean, dict):
        raise ValueError("public TD sidecar lacks public_support_plan support bounds")
    for key, expected in (("mass1", request["mass1"]), ("mass2", request["mass2"]), ("f_low", request["flow"])):
        if key not in clean or float(clean[key]) != expected:
            raise ValueError(f"public TD sidecar support plan does not match request: {key}={clean.get(key)!r}, expected {expected}")
    fields = {name: torch.tensor(float(clean[name]), dtype=torch.float64) for name in TDSupportPlan.__dataclass_fields__}
    return dt, TDSupportPlan(**fields)


def _epoch_counterfactual(freq: np.ndarray, value: np.ndarray, reference: np.ndarray, epoch_seconds: float | None) -> dict[str, Any]:
    if epoch_seconds is None or not math.isfinite(epoch_seconds):
        return {"status": "unavailable", "applied": False}
    factor = np.exp(-2j * np.pi * freq * epoch_seconds)
    return {"status": "diagnostic_only", "applied": False, "epoch_seconds": epoch_seconds, "factor_convention": "exp(-2 pi i f epoch_seconds)", "counterfactual_metrics": _fd_metrics(value * factor, reference)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="catalog", help="case directory name")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--public-td-npz", type=Path, default=None, help="optional cached exact-start public raw TD (requires --public-td-json)")
    parser.add_argument("--public-td-json", type=Path, default=None, help="sidecar proving the optional public TD request, support, dt, and sign")
    args = parser.parse_args()
    if (args.public_td_npz is None) != (args.public_td_json is None):
        parser.error("--public-td-npz and --public-td-json must be supplied together")

    case_dir = args.input_dir / args.case
    out_dir = args.output_dir or case_dir
    continuous_path = case_dir / "continuous.npz"
    public_fd_path = case_dir / "public-fd.npz"
    report_path = case_dir / "continuous.json"
    for path in (continuous_path, public_fd_path, report_path):
        if not path.exists():
            raise FileNotFoundError(f"required cache input is missing: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(continuous_path, allow_pickle=False) as native, np.load(public_fd_path, allow_pickle=False) as public_fd:
        report = json.loads(report_path.read_text())
        request = _request(report, native)
        raw_hp = _require(native, "h_plus_td")
        raw_hc = _require(native, "h_cross_td")
        cached_f = _require(native, "frequencies")
        cached_hp = _require(native, "h_plus_fd")
        cached_hc = _require(native, "h_cross_fd")
        replay = _condition(raw_hp, raw_hc, request)
        replay_f = replay.frequencies.detach().cpu().numpy()
        replay_hp = replay.h_plus.detach().cpu().numpy()
        replay_hc = replay.h_cross.detach().cpu().numpy()
        replay_check = {"frequency_exact": bool(np.array_equal(replay_f, cached_f)), "plus_exact": bool(np.array_equal(replay_hp, cached_hp)), "cross_exact": bool(np.array_equal(replay_hc, cached_hc)), "plus_max_abs": float(np.max(np.abs(replay_hp - cached_hp))), "cross_max_abs": float(np.max(np.abs(replay_hc - cached_hc)))}

        public_f = _require(public_fd, "frequencies")
        public_hp = _require(public_fd, "h_plus_fd")
        public_hc = _require(public_fd, "h_cross_fd")
        freq, ni, pi = _shared_bins(replay_f, public_f)
        band_mask = (freq >= request["flow"]) & (freq < request["fhigh"])
        freq = freq[band_mask]
        ni, pi = ni[band_mask], pi[band_mask]
        values = {"plus": replay_hp[ni], "cross": replay_hc[ni]}
        refs = {"plus": public_hp[pi], "cross": public_hc[pi]}
        metrics = {pol: {"raw": _fd_metrics(values[pol], refs[pol]), "frequency_bands": _band_metrics(freq, values[pol], refs[pol], request["flow"], request["fhigh"]), "linear_phase_ratio": _phase_ratio(freq, values[pol], refs[pol])} for pol in POLARIZATIONS}

        epoch = _scalar(native, "physical_epoch_seconds")
        if request["tc"] == 0.0:
            epoch_diagnostic = {pol: _epoch_counterfactual(freq, values[pol], refs[pol], epoch) for pol in POLARIZATIONS}
        else:
            epoch_diagnostic = {pol: {"status": "unavailable", "applied": False, "reason": "epoch counterfactual is defined only for tc=0"} for pol in POLARIZATIONS}
        td_public = None
        public_td_sidecar: dict[str, Any] | None = None
        public_conditioner = {"status": "not_run"}
        if args.public_td_npz is not None:
            public_td_sidecar = json.loads(args.public_td_json.read_text())
            actual_public_sha = _sha256(args.public_td_npz)
            declared_public_sha = public_td_sidecar.get("outputs", {}).get("public_td_sha256")
            if declared_public_sha is not None and declared_public_sha != actual_public_sha:
                raise ValueError("public TD sidecar hash does not match --public-td-npz")
            public_dt, public_support = _validate_public_td_sidecar(public_td_sidecar, request)
            td_public = np.load(args.public_td_npz, allow_pickle=False)
            public_raw_hp = _require(td_public, "exact_start_raw_h_plus", "h_plus_td")
            public_raw_hc = _require(td_public, "exact_start_raw_h_cross", "h_cross_td")
            declared_samples = public_td_sidecar.get("sampling", {}).get("samples")
            if declared_samples is not None and int(declared_samples) != int(public_raw_hp.size):
                raise ValueError("public TD sidecar sample count does not match --public-td-npz")
            public_replay = _condition(public_raw_hp, public_raw_hc, request, support=public_support, dt=public_dt)
            pf = public_replay.frequencies.detach().cpu().numpy()
            php = public_replay.h_plus.detach().cpu().numpy()
            phc = public_replay.h_cross.detach().cpu().numpy()
            si, sn, sp = _shared_bins(pf, public_f)
            pband = (si >= request["flow"]) & (si < request["fhigh"])
            public_conditioner = {"status": "complete", "input_sha256": _sha256(args.public_td_npz), "sidecar_sha256": _sha256(args.public_td_json), "dt_seconds": public_dt, "frequency_exact": bool(np.array_equal(pf, public_f)), "plus": _fd_metrics(php[sn[pband]], public_hp[sp[pband]]), "cross": _fd_metrics(phc[sn[pband]], public_hc[sp[pband]])}
        attachment_index = int(_scalar(native, "event_attachment_index") or 0)
        attachment_time = _scalar(native, "event_attachment_time")
        td_result, td_arrays = _td_control(native, td_public, attachment_time=attachment_time, cached_attachment_index=attachment_index)
        td_result["public_conditioner_fd"] = public_conditioner
        if td_public is not None:
            td_public.close()

        arrays: dict[str, np.ndarray] = {"shared_frequencies": freq, "replayed_h_plus_fd": replay_hp, "replayed_h_cross_fd": replay_hc, "shared_h_plus_fd": values["plus"], "shared_h_cross_fd": values["cross"], "public_h_plus_fd": refs["plus"], "public_h_cross_fd": refs["cross"], "residual_h_plus_fd": values["plus"] - refs["plus"], "residual_h_cross_fd": values["cross"] - refs["cross"], "ratio_h_plus_fd": np.divide(values["plus"], refs["plus"], out=np.full_like(values["plus"], np.nan + 1j * np.nan), where=np.abs(refs["plus"]) > 0), "ratio_h_cross_fd": np.divide(values["cross"], refs["cross"], out=np.full_like(values["cross"], np.nan + 1j * np.nan), where=np.abs(refs["cross"]) > 0)}
        arrays.update(td_arrays)
        np.savez_compressed(out_dir / "controls.npz", **arrays)

    output = {
        "status": "complete",
        "case": args.case,
        "inputs": {"continuous_npz": str(continuous_path), "continuous_sha256": _sha256(continuous_path), "public_fd_npz": str(public_fd_path), "public_fd_sha256": _sha256(public_fd_path), "continuous_json": str(report_path), "public_td_npz": str(args.public_td_npz) if args.public_td_npz else None, "public_td_sha256": _sha256(args.public_td_npz) if args.public_td_npz else None, "public_td_json": str(args.public_td_json) if args.public_td_json else None, "public_td_json_sha256": _sha256(args.public_td_json) if args.public_td_json else None},
        "request": request,
        "method": {"evolution": False, "oracle_fitting": False, "frequency_interpolation": False, "time_or_phase_alignment": False, "shared_frequency_selection": "exact bins only; score is [flow,fhigh)", "tc_runtime_value": request["tc"]},
        "native_replay": replay_check,
        "public_comparison": {"shared_bins": int(freq.size), "metrics": metrics, "epoch_factor_counterfactual": epoch_diagnostic},
        "td_control": td_result,
        "outputs": {"controls_npz": str(out_dir / "controls.npz")},
        "provenance": {"host": socket.gethostname(), "cwd": str(Path.cwd()), "pid": os.getpid(), "command": shlex.join([sys.executable, *sys.argv]), "script_sha256": _sha256(Path(__file__)), "native_conditioner_sha256": _sha256(ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py")},
        "limitations": ["The optional public TD control is absent unless --public-td-npz and its matching --public-td-json sidecar are supplied.", "The epoch factor is evaluated only as a counterfactual diagnostic at tc=0 and is never applied to the replay.", "Unequal TD lengths use shared elapsed indices and report unmatched-tail energy; no shift is fitted."],
    }
    (out_dir / "controls.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "case": args.case, "replay": replay_check, "shared_bins": int(freq.size), "controls": str(out_dir / "controls.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
