#!/usr/bin/env python3
"""Read-only sampled-versus-continuous peak gauge diagnostic.

This compares only the fresh SXS:BBH:2488 Lev5/N3 and Lev4/N3 NPZs.  The
``original`` branch copies the frozen strategy's sampled ``argmax(|h22|)``
gauge.  The ``continuous`` branch fits a quadratic to the three samples
around that maximum, linearly interpolates the complex h22/h33 values at the
quadratic vertex, and applies the same shared orbital rotation.  Both branches
then use the frozen complex FFT implementation and frequency grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.fit_nr33_amplitude_pilot import FFTControls
from scripts.nr33_phase_data import (
    interpolate_complex_common_grid,
    transform_complex,
)
from scripts.nr33_phase_metrics import phase_metrics


ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "docs/reviews/2026-09-15-parity-strategy-experiment"
MANIFEST = BASE / "data/manifest.json"
FRESH = BASE / "nr/fresh-holdout.json"
NPZ = {
    "Lev5/N3": Path("/tmp/torchwave-parity-strategy/npz/SXS_BBH_2488_8tb5f-rnc06_Lev5_N3.npz"),
    "Lev4/N3": Path("/tmp/torchwave-parity-strategy/npz/SXS_BBH_2488_8tb5f-rnc06_Lev4_N3.npz"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows() -> dict[str, dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text())
    wanted = {"SXS:BBH:2488/Lev5/N3": "Lev5/N3", "SXS:BBH:2488/Lev4/N3": "Lev4/N3"}
    out = {}
    for row in manifest["cases"]:
        if row["id"] in wanted:
            out[wanted[row["id"]]] = row
    if set(out) != set(NPZ):
        raise ValueError("manifest lacks both required fresh 2488 rows")
    for label, row in out.items():
        path = NPZ[label]
        if sha256(path) != row["npz_sha256"]:
            raise ValueError(f"NPZ hash mismatch for {label}")
    return out


def quadratic_peak(t: np.ndarray, h22: np.ndarray) -> tuple[int, float]:
    amp = np.abs(h22)
    i = int(np.argmax(amp))
    if i == 0 or i == len(t) - 1:
        raise ValueError("sampled peak must have two neighbors")
    # Center the abscissa before fitting: the local coordinate is O(dt), not
    # O(peak time), which avoids needless conditioning loss at t~9500.
    local_t = t[i - 1:i + 2] - t[i]
    coef = np.polyfit(local_t, amp[i - 1:i + 2], 2)
    if not np.isfinite(coef).all() or coef[0] >= 0:
        raise ValueError("local peak quadratic is not concave")
    vertex = float(t[i] - coef[1] / (2.0 * coef[0]))
    if not (t[i - 1] <= vertex <= t[i + 1]):
        raise ValueError("quadratic vertex left the three-sample bracket")
    return i, vertex


def linear_complex(t: np.ndarray, z: np.ndarray, x: float) -> complex:
    return complex(np.interp(x, t, z.real), np.interp(x, t, z.imag))


def gauge(t: np.ndarray, h22: np.ndarray, h33: np.ndarray, continuous: bool) -> tuple[np.ndarray, float, float, int]:
    i, tq = quadratic_peak(t, h22)
    peak = tq if continuous else float(t[i])
    if continuous:
        p22 = linear_complex(t, h22, peak)
        p33 = linear_complex(t, h33, peak)
    else:
        p22, p33 = h22[i], h33[i]
    orbital = float(np.angle(p22) / 2.0)
    rotated33_peak = p33 * np.exp(-3j * orbital)
    if abs(rotated33_peak.imag) < 1e-8 * abs(rotated33_peak):
        raise ValueError("orbital pi branch is ill-conditioned")
    branch = 0
    if rotated33_peak.imag < 0:
        orbital += np.pi
        branch = 1
    modes = np.stack((h22 * np.exp(-2j * orbital), h33 * np.exp(-3j * orbital)))
    return modes, peak, orbital, branch


def spectrum(row: dict[str, Any], continuous: bool, f: np.ndarray) -> dict[str, Any]:
    with np.load(NPZ["Lev5/N3" if row["resolution"] == "Lev5" else "Lev4/N3"], allow_pickle=False) as z:
        t = np.asarray(z["time"], float)
        h22 = np.asarray(z["h22"], complex)
        h33 = np.asarray(z["h33"], complex)
    modes, peak, orbital, branch = gauge(t, h22, h33, continuous)
    sample_index = int(np.argmax(np.abs(h22)))
    controls = FFTControls()
    out = []
    for mode in modes:
        transformed = transform_complex(
            t, mode, controls,
            relaxation_time=float(row["metadata"]["relaxation_time"]),
            h22_peak_time=peak,
        )
        out.append(interpolate_complex_common_grid(transformed.frequency_Mf, transformed.spectrum, f))
    return {"h": np.stack(out), "sample_peak": float(t[sample_index]),
            "peak": peak, "orbital": orbital, "branch": branch,
            "neighbor_spacing": [float(t[sample_index] - t[sample_index - 1]),
                                 float(t[sample_index + 1] - t[sample_index])],
            "n_samples": int(t.size), "time": [float(t[0]), float(t[-1])]}


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare(candidate: dict[str, Any], reference: dict[str, Any], f: np.ndarray) -> dict[str, Any]:
    result = {"modes": {}}
    for i, name in enumerate(("22", "33")):
        mask = np.abs(reference["h"][i]) >= 1e-3 * np.max(np.abs(reference["h"][i]))
        pm = phase_metrics(f, candidate["h"][i], reference["h"][i], phase_mask=mask)
        result["modes"][name] = {
            "complex_l2": relative_l2(candidate["h"][i], reference["h"][i]),
            "amplitude_l2": relative_l2(np.abs(candidate["h"][i]), np.abs(reference["h"][i])),
            "phase_fit": {k: pm[k] for k in ("constant_phase", "affine_phase", "time_shift", "raw_phase_rms", "constant_phase_rms", "affine_phase_rms")},
        }
    result["mean_complex_l2"] = float(np.mean([v["complex_l2"] for v in result["modes"].values()]))
    return result


def main(out_path: Path) -> None:
    metadata = rows()
    fresh = json.loads(FRESH.read_text())
    band = fresh["results"][0]["frequency_band"]
    f = float(band[0]) + np.arange(512, dtype=float) / 512.0 * (0.3 - float(band[0]))
    original = {label: spectrum(row, False, f) for label, row in metadata.items()}
    continuous = {label: spectrum(row, True, f) for label, row in metadata.items()}
    report: dict[str, Any] = {
        "purpose": "diagnose sampled h22 peak gauge; no model scoring, fitting, or refit",
        "inputs": {"manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
                   "fresh_holdout": str(FRESH), "fresh_holdout_sha256": sha256(FRESH),
                   "npz": {k: {"path": str(NPZ[k]), "sha256": sha256(NPZ[k])} for k in NPZ}},
        "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "method": {
            "original": "frozen sampled argmax(|h22|); h22/h33 values at sample; shared exp(-i*m*phi) rotation with frozen h33 pi branch",
            "continuous": "concave quadratic fit to three |h22| samples around argmax; linear complex interpolation of h22/h33 at vertex; same shared rotation and h33 pi branch",
            "fft": "scripts.nr33_phase_data.transform_complex with default FFTControls; same fresh-holdout 512-bin band",
            "frequency_band": [float(f[0]), float(f[-1])], "bins": int(f.size),
        },
        "cases": {},
    }
    for label in ("Lev5/N3", "Lev4/N3"):
        report["cases"][label] = {"original": {k: v for k, v in original[label].items() if k != "h"},
                                   "continuous": {k: v for k, v in continuous[label].items() if k != "h"},
                                   "peak_shift": float(continuous[label]["peak"] - original[label]["peak"]),
                                   "orbital_shift": float(continuous[label]["orbital"] - original[label]["orbital"])}
    report["peak_spacing"] = {
        "original_Lev5_minus_Lev4": float(original["Lev5/N3"]["peak"] - original["Lev4/N3"]["peak"]),
        "continuous_Lev5_minus_Lev4": float(continuous["Lev5/N3"]["peak"] - continuous["Lev4/N3"]["peak"]),
        "continuous_minus_original": float((continuous["Lev5/N3"]["peak"] - continuous["Lev4/N3"]["peak"]) - (original["Lev5/N3"]["peak"] - original["Lev4/N3"]["peak"])),
    }
    report["resolution_difference"] = {
        "original_sampled_gauge": compare(original["Lev4/N3"], original["Lev5/N3"], f),
        "continuous_quadratic_gauge": compare(continuous["Lev4/N3"], continuous["Lev5/N3"], f),
        "change_continuous_minus_original": {},
    }
    old = report["resolution_difference"]["original_sampled_gauge"]
    new = report["resolution_difference"]["continuous_quadratic_gauge"]
    for mode in ("22", "33"):
        report["resolution_difference"]["change_continuous_minus_original"][mode] = {
            "complex_l2": float(new["modes"][mode]["complex_l2"] - old["modes"][mode]["complex_l2"]),
            "amplitude_l2": float(new["modes"][mode]["amplitude_l2"] - old["modes"][mode]["amplitude_l2"]),
            "affine_phase_rms": float(new["modes"][mode]["phase_fit"]["affine_phase_rms"] - old["modes"][mode]["phase_fit"]["affine_phase_rms"]),
            "time_shift": float(new["modes"][mode]["phase_fit"]["time_shift"] - old["modes"][mode]["phase_fit"]["time_shift"]),
        }
    frozen = next(c for c in fresh["results"][0]["quality"]["comparisons"]
                  if c["control"] == "SXS:BBH:2488/Lev4/N3")
    checks = []
    for mode in ("22", "33"):
        checks += [
            abs(old["modes"][mode]["complex_l2"] - frozen["modes"][mode]["complex_l2"]),
            abs(old["modes"][mode]["phase_fit"]["time_shift"] - frozen["phase"][mode]["time_shift"]),
            abs(old["modes"][mode]["phase_fit"]["affine_phase_rms"] - frozen["phase"][mode]["affine_phase_rms"]),
        ]
    report["reproduction_check"] = {
        "comparison": "original sampled gauge versus fresh-holdout.json Lev4/N3 quality record",
        "max_absolute_difference": float(max(checks)), "pass_at_1e-12": bool(max(checks) <= 1e-12),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["resolution_difference"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("report.json"))
    main(parser.parse_args().output)
