#!/usr/bin/env python3
"""Frozen-input phase diagnostic; public LAL outputs are validation only.

No parameter calibration is performed. All complex spectra, including failed
quality diagnostics, are retained for replay and independent metric checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys

import numpy as np

from scripts.fit_nr33_amplitude_pilot import FFTControls, frequency_amplitude, independent_qnm33
from scripts.nr33_phase_data import load_complex_case, transform_complex, interpolate_complex_common_grid
from scripts.nr33_phase_metrics import phase_metrics, optimized_overlap
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC

PROTOCOL_SHA256 = "39b9fb1222aad2b54b0abac6f2d6caf089428111cdeea73b276eb2e9b8bd0774"
CALIBRATION_SHA256 = "ade377c62a57a5d9cff513e34f22b0fe43677177a0c1577a9a4af5ad3b8aa561"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_inputs(protocol_path, manifest_path, freeze_path):
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("phase protocol differs from preregistered hash")
    protocol, manifest, freeze = map(read_json, (protocol_path, manifest_path, freeze_path))
    if sha256(manifest_path) != protocol["input_manifest_sha256"]:
        raise ValueError("input manifest hash changed")
    if protocol["calibration_sha256"] != CALIBRATION_SHA256:
        raise ValueError("amplitude calibration freeze changed")
    for path, digest in {**manifest["source_sha256"], **freeze["source_sha256"]}.items():
        if sha256(path) != digest:
            raise ValueError(f"frozen source changed: {path}")
    # Fail if a supplied freeze omits a participating executable source.
    required = [Path(__file__).resolve(), Path(__file__).with_name("nr33_phase_data.py").resolve(),
                Path(__file__).with_name("nr33_phase_metrics.py").resolve(),
                Path(__file__).with_name("fit_nr33_amplitude_pilot.py").resolve(),
                Path(__file__).with_name("kerr_leaver_solver.py").resolve()]
    if not all(str(path) in freeze["source_sha256"] for path in required):
        raise ValueError("source freeze lacks participating code")
    if freeze["protocol_sha256"] != sha256(protocol_path) or freeze["input_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("source freeze does not bind protocol and inputs")
    primary_ids = manifest["primary_ids"]
    if primary_ids != protocol["primary_ids"] or len(set(primary_ids)) != 7:
        raise ValueError("primary selection changed")
    rows = manifest["cases"]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or len(rows) != 28:
        raise ValueError("source cases must be 28 distinct frozen rows")
    for row in rows:
        if sha256(row["path"]) != row["npz_sha256"]:
            raise ValueError(f"NPZ hash changed: {row['id']}")
    for cid in primary_ids:
        row = by_id[cid]
        if row["split"] != "holdout" or row["extrapolation_order"] != 3:
            raise ValueError("primary must remain an N3 holdout")
        siblings = [r for r in rows if r["sxs_id"] == row["sxs_id"]]
        levels = sorted({int(r["resolution"][3:]) for r in siblings})
        if len(siblings) != 4 or len(levels) != 2 or int(row["resolution"][3:]) != levels[-1]:
            raise ValueError("primary must be highest resolution of two N2/N3 pairs")
        if {(int(r["resolution"][3:]), r["extrapolation_order"]) for r in siblings} != {(l, n) for l in levels for n in (2, 3)}:
            raise ValueError("incomplete extraction/resolution controls")
    return protocol, manifest, freeze


def quadrature_weights(f):
    f = np.asarray(f, float)
    if f.ndim != 1 or len(f) < 8 or not np.all(np.isfinite(f)) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("need at least eight increasing positive frequency bins")
    widths = np.empty_like(f)
    widths[0] = (f[1] - f[0]) / 2
    widths[-1] = (f[-1] - f[-2]) / 2
    widths[1:-1] = (f[2:] - f[:-2]) / 2
    return widths


def phase_support(reference, relative_floor):
    amplitude = np.abs(reference)
    mask = amplitude >= float(relative_floor) * np.max(amplitude)
    # End at the first low-amplitude bin; never reconnect noisy tail islands.
    first_missing = np.flatnonzero(~mask)
    if first_missing.size:
        mask[first_missing[0]:] = False
    if np.count_nonzero(mask) < 8:
        raise ValueError("insufficient contiguous phase support")
    return mask


def lal_complex(f, q, total_mass, *, lal_module=None, ls_module=None):
    if lal_module is None or ls_module is None:
        import lal as lal_module
        import lalsimulation as ls_module
    mass_sec = total_mass * MSUN_SEC
    hz = np.asarray(f, float) / mass_sec
    vector = lal_module.CreateREAL8Vector(len(hz))
    vector.data[:] = hz
    params = lal_module.CreateDict()
    ls_module.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(params, 122022)
    release = int(ls_module.SimInspiralWaveformParamsLookupPhenomXHMReleaseVersion(params))
    if release != 122022:
        raise ValueError("LAL public release setter did not select 122022")
    distance = 100. * MPC_SI
    m2 = total_mass / (1 + q)
    series = ls_module.SimIMRPhenomXHMFrequencySequenceOneMode(
        vector, q * m2 * lal_module.MSUN_SI, m2 * lal_module.MSUN_SI,
        0., 0., 3, -3, distance, 0., 20., params)
    physical = np.asarray(series.data.data, dtype=np.complex128).copy()
    if physical.shape != hz.shape or not np.all(np.isfinite(physical)):
        raise ValueError("LAL returned invalid complex output")
    scale = mass_sec ** 2 * C_SI / distance
    metadata = {"api": "SimIMRPhenomXHMFrequencySequenceOneMode", "mode": [3, -3],
                "release": release, "total_mass_msun": total_mass, "q": q,
                "chi1": 0., "chi2": 0., "distance_mpc": 100., "phi_ref": 0., "f_ref_hz": 20.,
                "Mf_to_Hz_mass_seconds": mass_sec, "physical_to_geometric_divisor": scale,
                "lal_version": str(lal_module.__version__), "lalsimulation_version": str(ls_module.__version__),
                "series_epoch_seconds": float(series.epoch), "no_interpolation": True}
    return physical / scale, hz, metadata


def make_spectrum(row, controls):
    data = load_complex_case(row["path"], expected_sha256=row["npz_sha256"], metadata=row["metadata"])
    peak = float(data.time[np.argmax(np.abs(data.h22))])
    if not np.isclose(peak, row["metadata"]["peak_time_h22"], rtol=0, atol=1e-9):
        raise ValueError("manifest h22 peak differs from source samples")
    spectrum = transform_complex(data.time, data.mode, controls,
                                 relaxation_time=float(row["metadata"]["relaxation_time"]),
                                 h22_peak_time=peak)
    return spectrum, data


def benchmark(protocol, manifest, out):
    import lalsimulation as ls
    base_controls = FFTControls(**protocol["fft_controls"])
    by_id = {r["id"]: r for r in manifest["cases"]}
    results = []
    for index, cid in enumerate(manifest["primary_ids"]):
        row = by_id[cid]
        print(f"case {index + 1}/7 {cid}: NR controls", flush=True)
        primary, data = make_spectrum(row, base_controls)
        # Protect agreement with the frozen amplitude-only numerical pipeline.
        old_f, old_a = frequency_amplitude(data.time, data.mode, base_controls,
            relaxation_time=row["metadata"]["relaxation_time"], peak_time=row["metadata"]["peak_time_h22"])
        if not np.array_equal(old_f, primary.frequency_Mf) or not np.allclose(old_a, np.abs(primary.spectrum), rtol=5e-13, atol=1e-14):
            raise ValueError("complex transform changed frozen amplitude pipeline")
        spectra = {"primary": primary}
        control_rows = {}
        siblings = [r for r in manifest["cases"] if r["sxs_id"] == row["sxs_id"] and r["id"] != cid]
        for sibling in siblings:
            key = "control_" + sibling["resolution"] + "_N" + str(sibling["extrapolation_order"])
            spectra[key], _ = make_spectrum(sibling, base_controls)
            control_rows[key] = sibling
        for vindex, variant in enumerate(protocol["preprocessing_variants"]):
            controls = FFTControls(**{**asdict(base_controls), **variant})
            spectra[f"preprocessing_{vindex}"], _ = make_spectrum(row, controls)
        qnm = independent_qnm33(row["metadata"])
        lower = max(.95 * qnm["f_ring_33"], *(s.frequency_Mf[0] for s in spectra.values()))
        upper = min(.3, *(s.frequency_Mf[-1] for s in spectra.values()))
        f = primary.frequency_Mf[(primary.frequency_Mf >= lower) & (primary.frequency_Mf <= upper)]
        weights = quadrature_weights(f)
        # Fixed parity factor maps conjugated positive-m NR to h_(3,-3).
        arrays = {key: -interpolate_complex_common_grid(s.frequency_Mf, s.spectrum, f) for key, s in spectra.items()}
        reference = arrays["primary"]
        mask = phase_support(reference, protocol["phase_alignment"]["mask_min_relative_reference_amplitude"])
        controls_report = []
        limits = protocol["phase_quality_diagnostic_limits"]
        for key, values in arrays.items():
            if key == "primary":
                continue
            metrics = phase_metrics(f, values, reference, weights=weights, phase_mask=mask)
            variant = key.startswith("preprocessing")
            pmax = limits["preprocessing_phase_rms_rad" if variant else "extraction_resolution_phase_rms_rad"]
            lmax = limits["preprocessing_aligned_complex_l2" if variant else "extraction_resolution_aligned_complex_l2"]
            # Low-resolution N2 combines both effects; retain as an extra control.
            combined = not variant and control_rows[key]["resolution"] != row["resolution"] and control_rows[key]["extrapolation_order"] == 2
            passed = metrics["affine_phase_rms"] <= pmax and metrics["affine_relative_l2"] <= lmax
            controls_report.append({"key": key, "case_id": control_rows[key]["id"] if not variant else cid,
                "required": not combined, "passed": bool(passed), "phase_limit_rad": pmax,
                "complex_l2_limit": lmax, "metrics": metrics})
        quality_pass = all(c["passed"] for c in controls_report if c["required"])
        print(f"case {cid}: phase quality={quality_pass}; public LAL validation", flush=True)
        lal_reports = []
        for mass in protocol["overlap"]["total_masses_msun"]:
            values, hz, lal_meta = lal_complex(f, row["q"], mass)
            arrays[f"lal_M{int(mass)}"] = values
            arrays[f"frequency_Hz_M{int(mass)}"] = hz
            keep = hz >= protocol["overlap"]["minimum_frequency_hz"]
            if keep.sum() < 8:
                raise ValueError("insufficient detector-band support")
            psd = np.asarray([ls.SimNoisePSDaLIGOZeroDetHighPower(float(x)) for x in hz[keep]])
            if np.any(~np.isfinite(psd)) or np.any(psd <= 0):
                raise ValueError("PSD must be positive and finite")
            arrays[f"psd_M{int(mass)}"] = psd
            # Mf quadrature differs from Hz by a constant which cancels in normalized metrics.
            detector_weights = quadrature_weights(f[keep]) / psd
            detector_weights /= detector_weights.max()
            detector = optimized_overlap(f[keep], values[keep], reference[keep], weights=detector_weights,
                time_bounds=tuple(protocol["overlap"]["time_bounds_M"]))
            metrics = phase_metrics(f, values, reference, weights=weights, phase_mask=mask)
            white = optimized_overlap(f, values, reference, weights=weights,
                time_bounds=tuple(protocol["overlap"]["time_bounds_M"]))
            lal_reports.append({"total_mass_msun": mass, "metadata": lal_meta, "metrics": metrics,
                "unweighted_optimized": white, "detector_optimized": detector,
                "detector_band_Hz": [float(hz[keep][0]), float(hz[keep][-1])],
                "amplitude_relative_l2": float(np.linalg.norm(np.abs(values) - np.abs(reference)) / np.linalg.norm(np.abs(reference)))})
        arrays["frequency_Mf"], arrays["phase_mask"], arrays["quadrature_weights"] = f, mask, weights
        npz = out / (row["sxs_id"].replace(":", "_") + ".npz")
        with npz.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        result = {"case_id": cid, "sxs_id": row["sxs_id"], "q": row["q"], "target_q": row["target_q"],
            "split": row["split"], "qnm": qnm, "frequency_Mf_range": [float(f[0]), float(f[-1])],
            "bins": len(f), "phase_bins": int(mask.sum()),
            "phase_band_Mf": [float(f[mask][0]), float(f[mask][-1])],
            "phase_quality_pass": quality_pass, "numerical_controls": controls_report,
            "fft_metadata": {key: spectrum.metadata for key, spectrum in spectra.items()},
            "lal_comparisons": lal_reports, "arrays": {"path": str(npz.resolve()), "sha256": sha256(npz)},
            "frozen_amplitude_transform_reproduced": True}
        write_json(out / (row["sxs_id"].replace(":", "_") + ".json"), result)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, manifest, freeze = verify_inputs(args.protocol, args.manifest, args.source_freeze)
    args.output.mkdir(parents=True, exist_ok=False)
    start = utc()
    cases = benchmark(protocol, manifest, args.output)
    verify_inputs(args.protocol, args.manifest, args.source_freeze)
    write_json(args.output / "report.json", {"schema": "nr33-phase-benchmark-results-v1",
        "started_utc": start, "finished_utc": utc(), "protocol_sha256": sha256(args.protocol),
        "input_manifest_sha256": sha256(args.manifest), "source_freeze_sha256": sha256(args.source_freeze),
        "calibration_sha256_before_after": CALIBRATION_SHA256, "new_calibration": False,
        "runtime_promoted": False, "host": socket.gethostname(), "python": platform.python_version(),
        "command": sys.argv, "cases": cases})
    print(json.dumps({"completed_cases": len(cases), "phase_quality_passed": sum(c["phase_quality_pass"] for c in cases)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
