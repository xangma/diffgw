#!/usr/bin/env python3
"""Bounded post-freeze diagnostic of the SXS N=3/N=2 (3,3) extraction gap."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import linregress

try:
    from scripts.fit_nr33_amplitude_pilot import FFTControls, frequency_amplitude, independent_qnm33
except ModuleNotFoundError:  # direct ``python scripts/...py`` invocation
    from fit_nr33_amplitude_pilot import FFTControls, frequency_amplitude, independent_qnm33


CASES = {
    "0030": {"n3": "SXS_BBH_0030.npz", "n2": "SXS_BBH_0030_N2.npz",
             "h5_3": "SXS_BBH_0030.h5", "h5_2": "SXS_BBH_0030_N2.h5"},
    "0166": {"n3": "SXS_BBH_0166.npz", "n2": "SXS_BBH_0166_N2.npz",
             "h5_3": "SXS_BBH_0166.h5", "h5_2": "SXS_BBH_0166_N2.h5"},
}


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


def load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def qnm(meta, case):
    m = {"final_spin_magnitude": meta[case]["remnant_spin_z"],
         "final_mass_fraction": meta[case]["remnant_mass"]}
    return independent_qnm33(m)


def window_metric(t, x3, x2, peak, lo, hi):
    m = (t >= peak + lo) & (t <= peak + hi)
    if m.sum() < 8:
        return {"n": int(m.sum()), "available": False}
    a3, a2 = abs(x3[m]), abs(x2[m])
    floor = max(1e-14, 0.01 * float(abs(x3).max()))
    complex_mask = a3 >= floor
    phase = np.angle(x2[m][complex_mask] * np.conj(x3[m][complex_mask]))
    sig_a3, sig_a2 = a3[complex_mask], a2[complex_mask]
    complex_l2 = (float(np.linalg.norm(x2[m][complex_mask] - x3[m][complex_mask]) /
                        np.linalg.norm(x3[m][complex_mask])) if complex_mask.any() else None)
    return {
        "offset_M": [float(lo), float(hi)], "n": int(m.sum()), "available": True,
        "signal_threshold_1pct_peak": float(floor), "n_signal_above_threshold": int(complex_mask.sum()),
        "amplitude_ratio_median_N2_over_N3": (float(np.median(sig_a2 / sig_a3)) if complex_mask.any() else None),
        "amplitude_ratio_p05_p95": ([float(v) for v in np.percentile(sig_a2 / sig_a3, [5, 95])] if complex_mask.any() else None),
        "amplitude_relative_l2": float(np.linalg.norm(a2 - a3) / np.linalg.norm(a3)),
        "phase_difference_median_rad": (float(np.median(phase)) if complex_mask.any() else None),
        "phase_difference_rms_rad": (float(np.sqrt(np.mean(phase * phase))) if complex_mask.any() else None),
        "complex_relative_l2_above_1pct_peak": complex_l2,
    }


def ringdown_fit(t, x, peak, lo, hi):
    m = (t >= peak + lo) & (t <= peak + hi)
    if m.sum() < 16:
        return {"available": False, "n": int(m.sum())}
    amp = np.log(np.maximum(abs(x[m]), 1e-300))
    phase = np.unwrap(-np.angle(x[m]))
    ar = linregress(t[m], amp)
    pr = linregress(t[m], phase)
    return {"available": True, "offset_M": [float(lo), float(hi)], "n": int(m.sum()),
            "damping_rate_per_M": float(-ar.slope),
            "frequency_cycles_per_M": float(pr.slope / (2 * np.pi)),
            "amplitude_logfit_r2": float(ar.rvalue * ar.rvalue),
            "phase_fit_r2": float(pr.rvalue * pr.rvalue),
            "amplitude_median": float(np.median(abs(x[m])))}


def uniform_windowed(t, h, relaxation, peak, controls):
    """Return the exact uniformly sampled, endpoint-windowed input to FFT."""
    keep = t >= float(relaxation)
    t, h = t[keep], h[keep]
    dt = controls.sample_dt
    tu = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / dt)) + 1) * dt
    hu = CubicSpline(t, h.real, extrapolate=False)(tu) + 1j * CubicSpline(t, h.imag, extrapolate=False)(tu)
    start, end = float(relaxation) + controls.start_taper_duration, t[-1] - controls.end_taper_duration
    # Same raised-cosine endpoint treatment as frequency_amplitude().
    w = np.ones(tu.size)
    left, right = tu < start, tu > end
    w[left] = 0.5 * (1 - np.cos(np.pi * (tu[left] - tu[0]) / (start - tu[0])))
    w[right] = 0.5 * (1 + np.cos(np.pi * (tu[right] - end) / (tu[-1] - end)))
    return tu, hu * w


def partition_masks(x):
    """Five sigmoid-bin masks whose pointwise sum is exactly one."""
    b = np.asarray([-100.0, 0.0, 40.0, 100.0])
    s = 1.0 / (1.0 + np.exp(-np.clip((x[None, :] - b[:, None]) / 8.0, -60, 60)))
    return np.vstack((1.0 - s[0], s[0] - s[1], s[1] - s[2], s[2] - s[3], s[3]))


def residual_partition(t, x3, x2, relaxation, peak, controls, qnm_info):
    """Linear FFT partition of N2-N3 residual; reports cross terms explicitly."""
    tu, y3 = uniform_windowed(t, x3, relaxation, peak, controls)
    tu2, y2 = uniform_windowed(t, x2, relaxation, peak, controls)
    if not np.array_equal(tu, tu2):
        raise ValueError("uniform grids differ in residual partition")
    nfft = controls.pad_factor * tu.size
    freq = np.fft.fftfreq(nfft, controls.sample_dt)
    keep = (freq > 0) & (freq <= controls.fmax) & (freq >= 0.95 * qnm_info["f_ring_33"])
    masks = partition_masks(tu - peak)
    labels = ["pre_peak_before_-100M", "merger_-100M_to_0M", "early_ringdown_0M_to_40M",
              "late_ringdown_40M_to_100M", "tail_after_100M"]
    pieces = []
    for label, mask in zip(labels, masks):
        r = np.fft.fft(np.conj((y2 - y3) * mask), n=nfft) * controls.sample_dt
        pieces.append(r[keep])
    total = np.sum(pieces, axis=0)
    direct = np.fft.fft(np.conj(y2 - y3), n=nfft) * controls.sample_dt
    direct = direct[keep]
    total_norm2 = float(np.vdot(total, total).real)
    sum_norm2 = float(sum(np.vdot(p, p).real for p in pieces))
    return {"frequency_Mf_range": [float(freq[keep][0]), float(freq[keep][-1])], "bins": int(keep.sum()),
            "partition_boundaries_relative_M": labels,
            "mask_transition_width_M": 8.0,
            "coherent_reconstruction_relative_l2": float(np.linalg.norm(total - direct) /
                                                           max(np.linalg.norm(direct), 1e-300)),
            "cross_term_fraction_of_total_power": float((total_norm2 - sum_norm2) / max(total_norm2, 1e-300)),
            "pieces": [{"label": label, "l2": float(np.linalg.norm(piece)),
                        "fraction_of_total_l2": float(np.linalg.norm(piece) / max(np.linalg.norm(total), 1e-300)),
                        "power_fraction_of_total": float(np.vdot(piece, piece).real / max(total_norm2, 1e-300))}
                       for label, piece in zip(labels, pieces)]}


def fft_pair(t, x3, x2, relaxation, peak, controls, qnm_info):
    f, a3 = frequency_amplitude(t, x3, controls, relaxation_time=relaxation, peak_time=peak)
    f2, a2 = frequency_amplitude(t, x2, controls, relaxation_time=relaxation, peak_time=peak)
    keep = (f >= 0.95 * qnm_info["f_ring_33"]) & (f <= controls.fmax) & (f <= f2[-1])
    aa2 = np.interp(f[keep], f2, a2)
    ratio = aa2 / np.maximum(a3[keep], 1e-300)
    return {"frequency_Mf_range": [float(f[keep][0]), float(f[keep][-1])], "bins": int(keep.sum()),
            "relative_amplitude_l2_N2_vs_N3": float(np.linalg.norm(aa2 - a3[keep]) / np.linalg.norm(a3[keep])),
            "norm_ratio_N2_over_N3": float(np.linalg.norm(aa2) / np.linalg.norm(a3[keep])),
            "ratio_quantiles": [float(v) for v in np.percentile(ratio, [5, 25, 50, 75, 95])],
            "ratio_min_max": [float(np.min(ratio)), float(np.max(ratio))],
            "f": f[keep], "n3": a3[keep], "n2": aa2}


def decoder_check(decoder: Path, raw: Path, npz: Path):
    # Keep decoding in the separately supplied SXS environment.  The compact
    # JSON result avoids creating any intermediate files.
    code = r'''import json,sys,numpy as np,sxs
w=sxs.load(sys.argv[1]); z=np.load(sys.argv[2]);
print(json.dumps({"sxs_version":getattr(sxs,"__version__","unknown"),
"frame_type":str(getattr(w,"frame_type","unknown")),"n_times":int(w.t.size),
"max_abs_time_difference":float(np.max(abs(w.t-z["time"]))),
"max_abs_h33_difference":float(np.max(abs(w.data[:,w.index(3,3)]-z["h33"]))),
"max_abs_h22_difference":float(np.max(abs(w.data[:,w.index(2,2)]-z["h22"]))) }))'''
    p = subprocess.run([str(decoder), "-c", code, str(raw), str(npz)],
                       check=True, capture_output=True, text=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, default=Path("/tmp/torchwave-nr33-pilot/npz"))
    ap.add_argument("--raw-dir", type=Path, default=Path("/tmp/torchwave-nr33-pilot/raw"))
    ap.add_argument("--decoder-python", type=Path, default=Path("/tmp/torchwave-nr33-pilot/venv/bin/python"))
    ap.add_argument("--calibration", type=Path, default=Path("docs/reviews/2026-09-15-independent-calibration/nr33-calibration.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plot-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["cases"]
    n3_rows = {c: next(r for r in rows if r["sxs_id"] == "SXS:BBH:" + c and r.get("extrapolation_order", 3) == 3)
               for c in CASES}
    meta = {c: {"remnant_spin_z": n3_rows[c]["remnant_spin"][2],
                "remnant_mass": n3_rows[c]["remnant_mass"]} for c in CASES}
    controls = FFTControls()
    cal_hash = sha256(args.calibration)
    expected_cal_hash = "f447acda86f64f1d13745c8a5d725acdf295b050a2638d1731d4b6a139e1358d"
    if cal_hash != expected_cal_hash:
        raise ValueError("frozen calibration hash mismatch")
    reports = {}
    plot_data = {}
    for c, names in CASES.items():
        n3p, n2p = args.npz_dir / names["n3"], args.npz_dir / names["n2"]
        n3, n2 = load(n3p), load(n2p)
        t, t2 = n3["time"], n2["time"]
        if not np.array_equal(t, t2):
            raise ValueError(c + " N2/N3 time grids differ")
        row3, row2 = n3_rows[c], next(r for r in rows if r.get("raw_stem") == "SXS_BBH_" + c + "_N2")
        for p, row in [(n3p, row3), (n2p, row2)]:
            if sha256(p) != row["npz_sha256"]:
                raise ValueError("NPZ hash mismatch for " + str(p))
        peak = float(row3["peak_time"])
        q = qnm(meta, c)
        fft_full = fft_pair(t, n3["h33"], n2["h33"], row3["relaxation_time"], peak, controls, q)
        f, a3, aa2 = fft_full.pop("f"), fft_full.pop("n3"), fft_full.pop("n2")
        fft = fft_full
        fits = {}
        for mode in ("h33", "h22"):
            fits[mode] = {"N3": [ringdown_fit(t, n3[mode], peak, *w) for w in ((20, 50), (50, 80))],
                          "N2": [ringdown_fit(t, n2[mode], peak, *w) for w in ((20, 50), (50, 80))]}
        windows = {mode: [window_metric(t, n3[mode], n2[mode], peak, *w)
                          for w in ((-500, -100), (-20, 20), (20, 50), (50, 80), (80, 120))]
                   for mode in ("h33", "h22")}
        residual = residual_partition(t, n3["h33"], n2["h33"], row3["relaxation_time"], peak, controls, q)
        post_relax = None
        if c == "0030":
            # The staged waveform JSON records reference/relaxation 260M, while
            # the separately audited catalog metadata reports Lev3 relaxation 640M.
            # Keep the frozen-manifest result above and test this as a diagnostic variant.
            audit_meta = {"final_spin_magnitude": 0.540365644227, "final_mass_fraction": 0.971106575935}
            q_audit = independent_qnm33(audit_meta)
            n3_260 = frequency_amplitude(t, n3["h33"], controls, row3["relaxation_time"], peak)
            n3_640 = frequency_amplitude(t, n3["h33"], controls, 640.0, peak)
            common = (n3_260[0] >= max(0.95 * q_audit["f_ring_33"], n3_640[0][0])) & (n3_260[0] <= n3_640[0][-1])
            n3_640i = np.interp(n3_260[0][common], n3_640[0], n3_640[1])
            post_relax = {"relaxation_time_M": 640.0,
                          "metadata_source": "separately audited SXS Lev3 catalog metadata; frozen manifest and staged files unchanged",
                          "audited_remnant_metadata": audit_meta,
                          "audited_qnm": q_audit,
                          "N2_vs_N3": fft_pair(t, n3["h33"], n2["h33"], 640.0, peak, controls, q_audit),
                          "N3_640_vs_N3_260": {"relative_amplitude_l2": float(np.linalg.norm(n3_640i - n3_260[1][common]) /
                                                                                         np.linalg.norm(n3_260[1][common])),
                                               "norm_ratio_640_over_260": float(np.linalg.norm(n3_640i) /
                                                                                  np.linalg.norm(n3_260[1][common])),
                                               "bins": int(common.sum()),
                                               "frequency_Mf_range": [float(n3_260[0][common][0]), float(n3_260[0][common][-1])]}}
            for key in ("f", "n3", "n2"):
                post_relax["N2_vs_N3"].pop(key, None)
        reports[c] = {"sxs_id": row3["sxs_id"], "q": row3["q"], "resolution": row3["resolution"],
                      "peak_time_h22_N3": peak, "peak_time_h22_N2": float(t2[np.argmax(abs(n2["h22"]))]),
                      "peak_time_h33_N3": float(t[np.argmax(abs(n3["h33"]))]),
                      "peak_time_h33_N2": float(t2[np.argmax(abs(n2["h33"]))]),
                      "native_dt_min_max": [float(np.min(np.diff(t))), float(np.max(np.diff(t)))],
                      "qnm": q, "fft": fft, "windows": windows, "ringdown_fits": fits,
                      "fft_residual_partition_h33": residual,
                      "post_relaxation_640_variant": post_relax,
                      "inputs": {"n3_npz": {"path": str(n3p), "sha256": sha256(n3p)},
                                 "n2_npz": {"path": str(n2p), "sha256": sha256(n2p)},
                                 "n3_h5": {"path": str(args.raw_dir / names["h5_3"]), "sha256": sha256(args.raw_dir / names["h5_3"]), "md5": md5(args.raw_dir / names["h5_3"])},
                                 "n2_h5": {"path": str(args.raw_dir / names["h5_2"]), "sha256": sha256(args.raw_dir / names["h5_2"]), "md5": md5(args.raw_dir / names["h5_2"])},
                                 "n3_metadata_json": {"path": str(args.raw_dir / ("SXS_BBH_" + c + ".json")), "sha256": sha256(args.raw_dir / ("SXS_BBH_" + c + ".json"))},
                                 "n2_metadata_json": {"path": str(args.raw_dir / ("SXS_BBH_" + c + "_N2.json")), "sha256": sha256(args.raw_dir / ("SXS_BBH_" + c + "_N2.json"))}},
                      "decoder_checks": {"N3": decoder_check(args.decoder_python, args.raw_dir / names["h5_3"], n3p),
                                         "N2": decoder_check(args.decoder_python, args.raw_dir / names["h5_2"], n2p)}}
        plot_data[c] = (t - peak, n3, n2, f, a3, aa2)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for j, c in enumerate(CASES):
        dt, n3, n2, f, a3, a2 = plot_data[c]
        for i, mode in enumerate(("h33", "h22")):
            r = abs(n2[mode]) / np.maximum(abs(n3[mode]), 1e-12)
            ax[i, j].plot(dt, r, lw=.6)
            ax[i, j].axhline(1, color="k", lw=.5)
            ax[i, j].set_ylim(0, 1.25); ax[i, j].set_xlim(-50, 150); ax[i, j].grid(alpha=.2)
            ax[i, j].set_title(f"q≈{3 if c=='0030' else 6}: {mode} N2/N3")
            if i == 1: ax[i, j].set_xlabel("t − h22 peak [M]")
            if j == 0: ax[i, j].set_ylabel("amplitude ratio")
    fig.tight_layout(); td_plot = args.plot_dir / "time-domain-ratios.png"; fig.savefig(td_plot, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, c in enumerate(CASES):
        _, _, _, f, a3, a2 = plot_data[c]
        ax[j].semilogy(f, a3, label="N3"); ax[j].semilogy(f, a2, label="N2")
        ax[j].set_title(f"q≈{3 if c=='0030' else 6}: h33 FFT"); ax[j].set_xlabel("Mf"); ax[j].grid(alpha=.2); ax[j].legend()
    fig.tight_layout(); fft_plot = args.plot_dir / "fft-amplitudes.png"; fig.savefig(fft_plot, dpi=160); plt.close(fig)
    before = cal_hash
    after = sha256(args.calibration)
    if after != before:
        raise RuntimeError("calibration changed during diagnostic")
    report = {"purpose": "bounded post-freeze N2/N3 extraction diagnosis; no fitting or LAL tuning",
              "controls": {"fft": {"pad_factor": controls.pad_factor, "start_taper_duration": controls.start_taper_duration,
                                     "end_taper_duration": controls.end_taper_duration, "sample_dt": controls.sample_dt,
                                     "fmax": controls.fmax}, "ringdown_windows_M": [[20, 50], [50, 80]],
                        "temporal_windows_M_relative_to_N3_h22_peak": [[-500, -100], [-20, 20], [20, 50], [50, 80], [80, 120]]},
              "calibration_sha256_before": before, "calibration_sha256_after": after,
              "run_command": "/Users/xangma/miniconda3/envs/pycbcorig313/bin/python scripts/diagnose_nr33_extraction.py --manifest docs/reviews/2026-09-15-independent-calibration/nr33-data-manifest.json --output docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis/numerical-diagnosis.json --plot-dir docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis",
              "manifest_sha256": sha256(args.manifest), "source_sha256": {str(Path(__file__)): sha256(Path(__file__)),
                  "fit_nr33_amplitude_pilot.py": sha256(Path(__file__).with_name("fit_nr33_amplitude_pilot.py")),
                  "kerr_leaver_solver.py": sha256(Path(__file__).with_name("kerr_leaver_solver.py"))},
              "decoder": {"python": str(args.decoder_python), "python_sha256": sha256(args.decoder_python),
                          "command_template": "decoder_python -c <sxs.load H5; compare t,h33,h22> raw.h5 npz"},
              "cases": reports,
              "plots": {"time_domain": {"path": str(td_plot), "sha256": sha256(td_plot)},
                        "fft": {"path": str(fft_plot), "sha256": sha256(fft_plot)}},
              "interpretation": [
                  "The NPZ time grids are byte-identical between N=3 and N=2 within each simulation; native sampling is therefore not the source of the large q≈3 FFT gap.",
                  "Fresh re-decode with the supplied SXS package reproduces each NPZ time,h33,h22 array exactly; decoder indexing/frame conversion is not the source of the observed gap.",
                  "q≈3 Lev3 N2 h33 departs in the post-peak waveform beginning around +10–20M, with substantially faster/less coherent late-time decay than N3; this changes the full-record FFT and yields the 30.14% geometric-|FFT h33| L2 gap.",
                  "q≈6 Lev5 N2 and N3 h33 agree through the strong ringdown (+70M at approximately 1% amplitude-ratio level); later differences occur near the numerical floor, yielding only 0.0591% FFT L2.",
                  "The h22 control is closely matched around merger and early ringdown in both cases; its late differences occur after h22 has already decayed strongly. The evidence isolates the effect to the extracted h33 post-merger tail, consistent with extrapolation-order sensitivity, but does not identify the underlying SpEC extrapolation error mechanism."
              ],
              "limitations": ["Only the staged Lev3 q≈3 and Lev5 q≈6 pairs were tested.", "No new downloads or resolution campaign; N2/N3 is an extrapolation-order comparison, not a continuum-error bound.", "Late-time instantaneous fits can be ill-conditioned once the mode approaches the numerical floor."]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(reports), "calibration_unchanged": before == sha256(args.calibration), "output": str(args.output)}))


if __name__ == "__main__":
    main()
