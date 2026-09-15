#!/usr/bin/env python3
"""Bounded SXS q=3 resolution/extrapolation sensitivity diagnosis.

This is a post-freeze diagnostic.  It does not fit coefficients or call LAL:
all spectra use the frozen NR preprocessing from ``fit_nr33_amplitude_pilot``.
The raw download is deliberately restricted to q=3, Lev4/5 N2/N3 and Lev3 N4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts.fit_nr33_amplitude_pilot import FFTControls, frequency_amplitude


RECORD_DOI = "10.5281/zenodo.13167328"
RECORD_URL = "https://zenodo.org/records/13167328"
API_FILE = "https://zenodo.org/api/records/13167328/files/{}/content"
DEFAULT_RAW = Path("/tmp/torchwave-nr33-diagnosis-resolution/raw")
DEFAULT_NPZ = Path("/tmp/torchwave-nr33-diagnosis-resolution/npz")
DEFAULT_OUTPUT = Path("docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis/resolution-diagnosis.json")
DEFAULT_PLOT_DIR = Path("docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis")
FIXED_BAND = (0.118037824, 0.3)
Q3 = {
    "q": 2.999993377707023,
}
TARGETS = (
    ("Lev3", 2, "SXS_BBH_0030_N2"),
    ("Lev3", 3, "SXS_BBH_0030"),
    ("Lev3", 4, "SXS_BBH_0030_Lev3_N4"),
    ("Lev4", 2, "SXS_BBH_0030_Lev4_N2"),
    ("Lev4", 3, "SXS_BBH_0030_Lev4_N3"),
    ("Lev5", 2, "SXS_BBH_0030_Lev5_N2"),
    ("Lev5", 3, "SXS_BBH_0030_Lev5_N3"),
)


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    records = [r for r in data["records"] if r.get("doi") == RECORD_DOI]
    if len(records) != 1:
        raise ValueError(f"expected one inventory record for {RECORD_DOI}")
    out = {}
    for f in records[0]["files"]:
        out[f["key"]] = {"size": int(f["size"]), "md5": f["checksum"].split(":", 1)[-1]}
    return out


def check_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    md5 = digest(path, "md5")
    if size != expected["size"] or md5 != expected["md5"]:
        raise ValueError(f"inventory mismatch for {path}: size={size}, md5={md5}")
    return {"path": str(path), "size": size, "md5": md5, "sha256": digest(path)}


def check_sidecar(sidecar: dict[str, Any], raw_info: dict[str, Any], side_info: dict[str, Any]) -> dict[str, Any]:
    """Guard that each metadata sidecar validates the exact downloaded HDF5."""
    validation = sidecar.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("metadata sidecar has no validation object")
    if int(validation.get("h5_file_size", -1)) != raw_info["size"]:
        raise ValueError(f"sidecar size does not match HDF5 for {raw_info['path']}")
    if validation.get("md5sum") != raw_info["md5"]:
        raise ValueError(f"sidecar MD5 does not match HDF5 for {raw_info['path']}")
    if "n_times" not in validation:
        raise ValueError(f"sidecar has no decoded sample count for {raw_info['path']}")
    memory = sidecar.get("modifications", {}).get("add_memory", {})
    if not np.isfinite(float(memory.get("integration_start_time", np.nan))):
        raise ValueError(f"sidecar has no finite integration start for {raw_info['path']}")
    return {"h5_file_size_md5_match": True, "sidecar_sha256": side_info["sha256"],
            "validated_n_times": int(validation["n_times"])}


def fetch(url: str, path: Path, size: int) -> None:
    from urllib.request import urlopen
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        return
    with urlopen(url) as source, path.open("wb") as target:
        total = 0
        while block := source.read(1024 * 1024):
            total += len(block)
            if total > size:
                raise ValueError(f"download exceeded inventory size for {path}")
            target.write(block)
    if total != size:
        raise ValueError(f"download size {total} != inventory size {size} for {path}")


def decode(raw: Path, metadata: Path, out: Path, decode_python: Path) -> dict[str, Any]:
    """Call the existing prepare extractor in the pinned SXS environment."""
    case = {"sxs_id": "SXS:BBH:0030", "raw_stem": raw.stem}
    code = (
        "import json,sys; from pathlib import Path; "
        "from scripts.prepare_nr33_calibration_data import extract_case; "
        "print(json.dumps(extract_case(json.loads(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([str(decode_python), "-c", code, json.dumps(case), str(raw.parent), str(out.parent)],
                          check=True, capture_output=True, text=True, env=env)
    # extract_case writes raw-stem.npz.  Preserve its provenance output, while
    # tolerating the pinned extractor's incidental stdout formatting.
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {"npz": str(out)}


def load_wave(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(row["npz"], allow_pickle=False) as data:
        t = np.asarray(data["time"], float)
        h = np.asarray(data["h33"], complex)
        h22 = np.asarray(data["h22"], complex) if "h22" in data else np.zeros_like(h)
    peak = float(t[int(np.argmax(np.abs(h22)))]) if np.any(h22) else float("nan")
    return t, h, h22, {"peak_time": peak}


def common_curve(rows: list[dict[str, Any]], controls: FFTControls, relaxation_key: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves = {}
    for row in rows:
        t, h, _, _ = load_wave(row)
        f, a = frequency_amplitude(t, h, controls, row[relaxation_key])
        eta = Q3["q"] / (1 + Q3["q"]) ** 2
        a022 = np.pi * np.sqrt(2 * eta / 3) * (np.pi * f) ** (-7 / 6)
        curves[row["id"]] = (f, a, a / a022)
    return curves


def compare(curve_a: tuple[np.ndarray, np.ndarray, np.ndarray], curve_b: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
    fa, aa, na = curve_a
    fb, ab, nb = curve_b
    lo = max(FIXED_BAND[0], float(fa[0]), float(fb[0]))
    hi = min(FIXED_BAND[1], float(fa[-1]), float(fb[-1]))
    keep = (fa >= lo) & (fa <= hi)
    if keep.sum() < 8:
        raise ValueError("insufficient common fixed-band FFT bins")
    f = fa[keep]
    b = np.interp(f, fb, ab)
    bn = np.interp(f, fb, nb)
    a, an = aa[keep], na[keep]
    return {
        "frequency_Mf_range": [float(f[0]), float(f[-1])], "bins": int(f.size),
        "geometric_relative_l2": float(np.linalg.norm(b - a) / np.linalg.norm(a)),
        "geometric_norm_ratio": float(np.linalg.norm(b) / np.linalg.norm(a)),
        "A022_normalized_relative_l2": float(np.linalg.norm(bn - an) / np.linalg.norm(an)),
        "A022_normalized_norm_ratio": float(np.linalg.norm(bn) / np.linalg.norm(an)),
    }


def time_compare(a: dict[str, Any], b: dict[str, Any], relaxation: float) -> dict[str, Any]:
    ta, ha, _, _ = load_wave(a)
    tb, hb, _, _ = load_wave(b)
    lo, hi = max(ta[0], tb[0], relaxation), min(ta[-1], tb[-1])
    keep = (ta >= lo) & (ta <= hi)
    if keep.sum() < 32:
        raise ValueError("insufficient common time interval")
    hb_i = np.interp(ta[keep], tb, hb.real) + 1j * np.interp(ta[keep], tb, hb.imag)
    h = ha[keep]
    return {
        "common_time": [float(ta[keep][0]), float(ta[keep][-1])], "samples": int(keep.sum()),
        "complex_relative_l2": float(np.linalg.norm(hb_i - h) / np.linalg.norm(h)),
        "amplitude_relative_l2": float(np.linalg.norm(np.abs(hb_i) - np.abs(h)) / np.linalg.norm(np.abs(h))),
        "peak_time_delta_M": float(b["peak_time"] - a["peak_time"]),
        "peak_amplitude_ratio": float(b["peak_amplitude"] / a["peak_amplitude"]),
    }


def plot(rows: list[dict[str, Any]], curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], out_dir: Path) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    amp_path = out_dir / "resolution-q3-amplitude-spectra.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in rows:
        f, a, _ = curves[row["id"]]
        k = (f >= FIXED_BAND[0]) & (f <= FIXED_BAND[1])
        ax.semilogy(f[k], a[k], label=row["short"])
    ax.set(xlabel="Dimensionless frequency Mf", ylabel="Geometric Fourier amplitude |h33|", title="SXS:BBH:0030 resolution/extrapolation")
    ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(amp_path, dpi=160); plt.close(fig)
    td_path = out_dir / "resolution-q3-time-domain-differences.png"
    ref = next(r for r in rows if r["id"] == "SXS:BBH:0030/Lev3/N3")
    tr, hr, _, _ = load_wave(ref)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in rows:
        if row is ref: continue
        t, h, _, _ = load_wave(row)
        lo, hi = max(tr[0], t[0], ref["native_relaxation_time"]), min(tr[-1], t[-1])
        k = (tr >= lo) & (tr <= hi)
        hi_i = np.interp(tr[k], t, h.real) + 1j * np.interp(tr[k], t, h.imag)
        ax.plot(tr[k], np.abs(hi_i - hr[k]) / np.maximum(np.abs(hr[k]), 1e-30), label=row["short"])
    ax.set(xlabel="Time M", ylabel="Pointwise relative |Δh33|", title="Relative to Lev3 N3"); ax.set_yscale("log"); ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(td_path, dpi=160); plt.close(fig)
    return [{"path": str(p), "sha256": digest(p)} for p in (amp_path, td_path)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--inventory", type=Path, default=Path("docs/reviews/2026-09-15-independent-calibration/verification/dataset-license-records.json"))
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    ap.add_argument("--decode-python", type=Path, default=Path("/tmp/torchwave-nr33-pilot/venv/bin/python"))
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    inv = inventory(args.inventory)
    rows = []
    manifest_rows = {r["id"]: r for r in json.loads(Path("docs/reviews/2026-09-15-independent-calibration/nr33-data-manifest.json").read_text())["cases"]}
    manifest_case_checks = {}
    extraction = {}
    for lev, order, stem in TARGETS:
        key = f"{lev}:Strain_N{order}.h5"; jkey = f"{lev}:Strain_N{order}.json"
        if key not in inv or jkey not in inv:
            raise ValueError(f"missing recorded inventory entries for {key}")
        pilot_stem = {("Lev3", 2): "SXS_BBH_0030_N2", ("Lev3", 3): "SXS_BBH_0030"}.get((lev, order))
        raw = (Path("/tmp/torchwave-nr33-pilot/raw") / f"{pilot_stem}.h5" if pilot_stem else args.raw_dir / f"{stem}.h5")
        side = (Path("/tmp/torchwave-nr33-pilot/raw") / f"{pilot_stem}.json" if pilot_stem else args.raw_dir / f"{stem}.json")
        npz = (Path("/tmp/torchwave-nr33-pilot/npz") / f"{pilot_stem}.npz" if pilot_stem else args.npz_dir / f"{stem}.npz")
        if args.download and pilot_stem is None:
            fetch(API_FILE.format(key), raw, inv[key]["size"])
            fetch(API_FILE.format(jkey), side, inv[jkey]["size"])
        raw_info = check_file(raw, inv[key]); side_info = check_file(side, inv[jkey])
        sidecar = json.loads(side.read_text())
        sidecar_guard = check_sidecar(sidecar, raw_info, side_info)
        if not npz.exists():
            extraction[stem] = decode(raw, side, npz, args.decode_python)
        t, h, h22, peak = load_wave({"npz": str(npz)})
        if int(sidecar["validation"]["n_times"]) != int(t.size):
            raise ValueError(f"sidecar decoded sample count does not match NPZ for {raw}")
        native_relaxation = float(sidecar["modifications"]["add_memory"]["integration_start_time"])
        row = {"id": f"SXS:BBH:0030/{lev}/N{order}", "short": f"{lev} N{order}", "resolution": lev, "extrapolation_order": order,
               "raw": raw_info, "metadata": side_info, "npz": str(npz), "npz_sha256": digest(npz),
               "n_times": int(t.size), "time_range": [float(t[0]), float(t[-1])], "peak_time": peak["peak_time"],
               "peak_amplitude": float(np.max(np.abs(h22))), "native_relaxation_time": native_relaxation,
               "frozen_manifest_relaxation_time": 260.0,
               "sidecar_metadata": {"integration_start_time": native_relaxation, "validation": sidecar.get("validation", {}),
                                    "guard": sidecar_guard}}
        if row["id"] in manifest_rows:
            m = manifest_rows[row["id"]]
            if abs(float(m["q"]) - Q3["q"]) > 1e-8 or m.get("raw_h5_md5") != raw_info["md5"] or m.get("npz_sha256") != row["npz_sha256"]:
                raise ValueError(f"manifest provenance mismatch for {row['id']}")
            manifest_case_checks[row["id"]] = {"q_match": True, "raw_h5_md5_match": True,
                                                 "npz_sha256_match": True, "manifest_path": m.get("path"),
                                                 "manifest_raw_h5": m.get("raw_h5")}
        rows.append(row)
    controls = FFTControls()
    curves = common_curve(rows, controls, "native_relaxation_time")
    frozen_curves = common_curve(rows, controls, "frozen_manifest_relaxation_time")
    by_id = {r["id"]: r for r in rows}
    per_resolution = []
    for lev in ("Lev3", "Lev4", "Lev5"):
        n2, n3 = by_id[f"SXS:BBH:0030/{lev}/N2"], by_id[f"SXS:BBH:0030/{lev}/N3"]
        native_amp = compare(curves[n3["id"]], curves[n2["id"]])
        frozen_amp = compare(frozen_curves[n3["id"]], frozen_curves[n2["id"]])
        per_resolution.append({"resolution": lev, "native_relaxation_time_M": n3["native_relaxation_time"],
                               "N2_vs_N3_amplitude": native_amp,
                               "N2_vs_N3_time": time_compare(n3, n2, n3["native_relaxation_time"]),
                               "frozen_manifest_260M_control": {"N2_vs_N3_amplitude": frozen_amp,
                                                                  "N2_vs_N3_time": time_compare(n3, n2, 260.0)},
                               "native_minus_frozen_geometric_relative_l2": float(native_amp["geometric_relative_l2"] - frozen_amp["geometric_relative_l2"])})
    n3 = by_id["SXS:BBH:0030/Lev3/N3"]; n4 = by_id["SXS:BBH:0030/Lev3/N4"]
    cross = []
    for row in rows:
        if row is n3: continue
        cross.append({"case_id": row["id"], "vs_Lev3_N3": compare(curves[n3["id"]], curves[row["id"]])})
    plots = plot(rows, curves, args.plot_dir)
    controls_dict = {"pad_factor": controls.pad_factor, "start_taper_duration": controls.start_taper_duration,
                     "end_taper_duration": controls.end_taper_duration, "fmax": controls.fmax,
                     "sample_dt": controls.sample_dt}
    out = {"purpose": "bounded q=3 NR resolution/extrapolation diagnosis; no fitting or LAL output",
           "record": {"doi": RECORD_DOI, "url": RECORD_URL, "inventory": str(args.inventory), "inventory_sha256": digest(args.inventory),
                      "manifest": "docs/reviews/2026-09-15-independent-calibration/nr33-data-manifest.json",
                      "manifest_sha256": digest(Path("docs/reviews/2026-09-15-independent-calibration/nr33-data-manifest.json")),
                      "metadata_audit": "docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis/metadata-consistency.json",
                      "metadata_audit_sha256": digest(Path("docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis/metadata-consistency.json"))},
           "manifest_case_checks": manifest_case_checks,
           "scope": {"sxs_id": "SXS:BBH:0030", "targets": [r["id"] for r in rows], "downloaded_target_count": 5,
                     "existing_pilot_target_count": 2, "fixed_frequency_band_Mf": list(FIXED_BAND)},
           "controls": {**controls_dict,
                        "preprocessing_source": "scripts/fit_nr33_amplitude_pilot.py", "A022_normalization": "Eq.26; eta=q/(1+q)^2",
                        "geometric_metric": "main reported amplitude metric; dt*FFT |conj(h33)|", "interpolation": "linear onto baseline curve bins within fixed common band"},
           "cases": rows, "per_resolution": per_resolution, "cross_resolution_vs_Lev3_N3": cross, "plots": plots,
           "provenance": {"diagnostic_script_sha256": digest(Path(__file__)), "fitter_sha256": digest(Path(__file__).with_name("fit_nr33_amplitude_pilot.py")),
                          "prepare_sha256": digest(Path(__file__).with_name("prepare_nr33_calibration_data.py")), "command": sys.argv,
                          "analysis_python": sys.executable, "decode_python": str(args.decode_python), "decoded": extraction},
           "metadata": {"native_relaxation_source": "exact Strain_N#.json sidecar modifications.add_memory.integration_start_time",
                        "lev3_exact_metadata_audit": {"q": 2.999951112567489, "reference_time": 640.0,
                                                       "remnant_mass": 0.971106575935, "remnant_spinz": 0.540365644227},
                        "lev4_exact_metadata_audit": {"reference_time": 280.0, "remnant_mass": 0.971162700375,
                                                       "remnant_spinz": 0.540686289841},
                        "lev5_exact_metadata_audit": {"q": 2.999993377707023, "reference_time": 260.0,
                                                       "remnant_mass": 0.971212971306, "remnant_spinz": 0.5406441997},
                        "frozen_manifest_mismatch": "manifest relaxation/reference 260M and remnant fields match Lev5 metadata, not Lev3; fixed-band geometric comparison does not use remnant fields",
                        "source_deprecation": "metadata audit marks SXS:BBH:0030 deprecated; superseded by SXS:BBH:1221, 2265, 2498"},
           "interpretation": {"Lev3_choice": "The geometric N2/N3 gap falls from about 30.16% (Lev3) to 15.54% (Lev4) to 3.40% (Lev5), so the original Lev3 choice is a supported artifact diagnosis for this extraction metric.",
                              "N4": "Lev3 N4 is substantially worse versus Lev3 N3 (about 111.88% geometric relative L2); it is an extrapolation sensitivity check, not a resolution-convergence proof.",
                              "limits": ["Only one q=3 simulation; extrapolation orders and resolutions are not independent simulations", "linear spectral interpolation and shared preprocessing can hide correlated errors", "time-domain comparison is interpolated onto the first waveform time grid", "the deprecated 0030 record is retained because it is the exact recorded source requested"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "targets": len(rows), "plots": len(plots)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
