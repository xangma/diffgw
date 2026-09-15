"""Compare a serialized clean-room EOB mode to public LAL TD output."""
import argparse
import datetime
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path

import lal
import lalsimulation as ls
import numpy as np
import torch


MTSUN = 4.925490947641267e-6


def invert(hp, hc, cos_i):
    """Invert the published nonprecessing (2,2) polarization projection."""
    return (math.sqrt(16 * math.pi / 5) / (1 + cos_i**2) * hp
            - 1j * math.sqrt(4 * math.pi / 5) / cos_i * hc)


def project(z, cos_i):
    """Project a 22 mode back to the plus/cross polarizations."""
    hp = math.sqrt(5 / (16 * math.pi)) * (1 + cos_i**2) * z.real
    hc = -math.sqrt(5 / (4 * math.pi)) * cos_i * z.imag
    return hp, hc


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interp_local(x, y, q, degree):
    n = degree + 1
    out = np.empty(q.shape, dtype=y.dtype)
    for k, value in enumerate(q):
        j = int(np.searchsorted(x, value))
        left = max(0, min(len(x) - n, j - n // 2))
        xx = x[left:left + n]
        center = xx[n // 2]
        scale = max(abs(xx[-1] - xx[0]), np.finfo(float).eps)
        u = (xx - center) / scale
        out[k] = np.polynomial.polynomial.polyval(
            (value - center) / scale,
            np.polynomial.polynomial.polyfit(u, y[left:left + n], degree))
    return out


def interpolate(x, y, q, kind):
    if kind == "linear":
        return np.interp(q, x, y.real) + 1j * np.interp(q, x, y.imag)
    return interp_local(x, y, q, 3 if kind == "cubic" else 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--time-origin", choices=("orbital_peak", "nqc_match"),
                    default="orbital_peak")
    ap.add_argument("--interpolation", choices=("linear", "cubic", "quintic"), default="linear")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = args.input_dir / "full20.pt"
    run_result_path = args.input_dir / "result.json"
    run_result = json.loads(run_result_path.read_text()) if run_result_path.exists() else {}
    d = torch.load(pt_path, weights_only=True)
    mass = 50 * MTSUN
    cos_i = math.cos(0.8)
    distance = 100e6 * lal.PC_SI
    hp, hc = ls.SimInspiralChooseTDWaveform(
        30 * lal.MSUN_SI, 20 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0,
        distance, 0.8, 0, 0, 0, 0, 0.5 * mass, 20, 0, lal.CreateDict(),
        ls.GetApproximantFromString("SEOBNRv4"))
    hpv = np.asarray(hp.data.data)
    hcv = np.asarray(hc.data.data)
    assert np.isfinite(hpv).all() and np.isfinite(hcv).all()
    tb_abs = float(hp.epoch) + np.arange(len(hpv)) * float(hp.deltaT)
    zb = invert(hpv, hcv, cos_i)
    # Keep LAL's public epoch coordinates; the native event subtraction below
    # places its selected event on the corresponding physical chart.
    tb = tb_abs
    reproj_p, reproj_c = project(zb, cos_i)
    proj_p_err = np.linalg.norm(reproj_p - hpv) / np.linalg.norm(hpv)
    proj_c_err = np.linalg.norm(reproj_c - hcv) / np.linalg.norm(hcv)
    assert proj_p_err < 1e-12 and proj_c_err < 1e-12

    t0_key = "peak_time" if args.time_origin == "orbital_peak" else "match_time"
    t0 = float(d[t0_key])
    tn = (np.asarray(d["time"].numpy()) - t0) * mass
    zn = (np.asarray(d["mode"].numpy()) * mass * 299792458
          / (100 * 3.085677581491367e22))
    for arr in (tn, zn, zb):
        assert np.isfinite(arr).all()
    assert np.all(np.diff(tn) > 0) and np.all(np.diff(tb) > 0)
    lo, hi = max(tn[0], tb[0]), min(tn[-1], tb[-1])
    tq = np.arange(lo, hi + 0.25e-9, 0.5 * mass)
    tq = tq[tq <= hi]
    assert len(tq) and tq[0] >= lo and tq[-1] <= hi
    an = interpolate(tn, zn, tq, args.interpolation)
    bn = interpolate(tb, zb, tq, args.interpolation)
    raw_l2 = np.linalg.norm(an - bn) / np.linalg.norm(bn)
    active = ((abs(an) > .001 * max(abs(an))) &
              (abs(bn) > .001 * max(abs(bn))))
    pa = np.unwrap(np.angle(an[active]))
    pb = np.unwrap(np.angle(bn[active]))
    phase = np.angle(an[active] * np.conj(bn[active]))
    tt = tq[active]
    fn = -np.gradient(pa, tt) / (2 * math.pi)
    fb = -np.gradient(pb, tt) / (2 * math.pi)
    event_native = np.interp(0, tn, zn.real) + 1j * np.interp(0, tn, zn.imag)
    event_binary = np.interp(0, tb, zb.real) + 1j * np.interp(0, tb, zb.imag)
    def segment_energy(a, b):
        if b <= a:
            return 0.0
        inside = (tb >= a) & (tb <= b)
        ts = np.concatenate(([a], tb[inside], [b]))
        zs = np.interp(ts, tb, zb.real) + 1j * np.interp(ts, tb, zb.imag)
        return float(np.trapezoid(abs(zs) ** 2, ts))
    total_binary_energy = float(np.trapezoid(abs(zb) ** 2, tb))
    leading_energy = segment_energy(tb[0], min(tn[0], tb[-1]))
    trailing_energy = segment_energy(max(tn[-1], tb[0]), tb[-1])
    tail_fraction = trailing_energy / total_binary_energy
    intervals = {}
    for name, mask in (("early", tq < -.1), ("inspiral", tq < 0),
                       ("merger", abs(tq) <= .02), ("ringdown", tq > .02)):
        if mask.any():
            intervals[name] = {
                "samples": int(mask.sum()),
                "relative_l2": float(np.linalg.norm(an[mask] - bn[mask]) /
                                      np.linalg.norm(bn[mask])),
                "phase_frequency_active_samples": int((active & mask).sum()),
            }
    out = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(), "torch": torch.__version__,
        "lal_version": getattr(lal, "__version__", "unknown"),
        "lalsimulation_version": getattr(ls, "__version__", "unknown"),
        "verifier_sha256": sha256(Path(__file__)), "input_sha256": sha256(pt_path),
        "run_result_sha256": sha256(run_result_path) if run_result_path.exists() else None,
        "run_source_hashes": run_result.get("source_sha256"),
        "run_cli": sys.argv, "interpolation": args.interpolation,
        "time_origin": args.time_origin, "time_origin_key": t0_key,
        "native_support_s": [float(tn[0]), float(tn[-1])],
        "binary_support_s": [float(tb[0]), float(tb[-1])],
        "common_support_s": [float(lo), float(hi)],
        "raw_common_relative_l2": float(raw_l2),
        "active_samples": int(active.sum()),
        "active_threshold": "amplitude > 0.1% of each waveform maximum; phase/frequency only",
        "native_frequency_hz": [float(fn.min()), float(fn.max())],
        "binary_frequency_hz": [float(fb.min()), float(fb.max())],
        "frequency_rms_hz": float(np.sqrt(np.mean((fn - fb) ** 2))),
        "phase_rms_rad": float(np.sqrt(np.mean(phase ** 2))),
        "phase_max_abs_rad": float(max(abs(phase))),
        "phase_residual_at_common_start_rad": float(phase[0]),
        "native_phase_at_same_event_rad": float(np.angle(event_native)),
        "binary_phase_at_same_event_rad": float(np.angle(event_binary)),
        "native_amplitude_peak_time_s": float(tn[np.argmax(abs(zn))]),
        "binary_amplitude_peak_time_s": float(tb[np.argmax(abs(zb))]),
        "dropped_tail_energy_fraction": tail_fraction,
        "dropped_tail_definition": "sum |binary h22|^2 after native end / total binary sum",
        "omitted_binary_energy": {
            "leading_integral": leading_energy, "trailing_integral": trailing_energy,
            "total_integral": total_binary_energy,
            "leading_fraction": leading_energy / total_binary_energy,
            "trailing_fraction": trailing_energy / total_binary_energy,
            "definition": "trapezoidal time integral outside common native/binary support",
        },
        "projection_roundtrip_relative_l2": {
            "plus": float(proj_p_err), "cross": float(proj_c_err),
        },
        "all_input_arrays_finite": True, "intervals": intervals,
    }
    shutil.copy2(pt_path, args.output_dir / "full20.pt")
    (args.output_dir / "result.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
