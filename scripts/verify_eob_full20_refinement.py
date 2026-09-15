"""Compare saved full20 payloads without rerunning the EOB driver."""
import argparse
import datetime
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch


MTSUN = 4.925490947641267e-6


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interp_local(x, y, q, degree):
    """Local polynomial interpolation, preserving complex values."""
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


def payload(path):
    d = torch.load(path, weights_only=True)
    keys = ("time", "mode", "trajectory_time", "state", "omega", "valid")
    a = {k: np.asarray(d[k].numpy()) for k in keys}
    assert all(np.isfinite(a[k]).all() for k in keys if k != "valid")
    assert a["valid"].dtype == bool and a["valid"].all()
    peak, match = float(d["peak_time"]), float(d["match_time"])
    return {"path": str(path), **a, "peak_time_M": peak, "match_time_M": match,
            "mode_peak_time_M": float(a["time"][np.argmax(abs(a["mode"]))]),
            "mode_peak_amplitude": float(abs(a["mode"]).max()), "sha256": sha256(path)}


def relative_mode(p, origin):
    t0 = 0.0 if origin == "initial" else p["peak_time_M"]
    return (p["time"] - t0) * 50 * MTSUN, p["mode"]


def trajectory_metrics(c, f, origin):
    def tr(p):
        t0 = 0.0 if origin == "initial" else p["peak_time_M"]
        return (p["trajectory_time"] - t0) * 50 * MTSUN
    tc, tf = tr(c), tr(f)
    assert np.all(np.diff(tc) > 0) and np.all(np.diff(tf) > 0)
    lo, hi = max(tc[0], tf[0]), min(tc[-1], tf[-1])
    dt = min(np.median(np.diff(tc)), np.median(np.diff(tf)))
    q = np.arange(lo, hi + dt * .25, dt); q = q[q <= hi]
    assert len(q) and q[-1] <= hi
    sc = np.column_stack([interp_local(tc, c["state"][:, j], q, 5).real
                          for j in range(c["state"].shape[1])])
    sf = np.column_stack([interp_local(tf, f["state"][:, j], q, 5).real
                          for j in range(f["state"].shape[1])])
    oc = interp_local(tc, c["omega"].astype(complex), q, 5).real
    of = interp_local(tf, f["omega"].astype(complex), q, 5).real
    assert np.isfinite(sc).all() and np.isfinite(sf).all()
    assert np.isfinite(oc).all() and np.isfinite(of).all()
    abs_cols = np.sqrt(np.mean((sc - sf) ** 2, axis=0))
    rel_cols = abs_cols / np.sqrt(np.mean(sf ** 2, axis=0))
    return {"common_support_s": [float(lo), float(hi)], "samples": int(len(q)),
            "state_columns": ["r", "phi", "prstar", "p_phi"],
            "state_absolute_l2_by_column": abs_cols.tolist(),
            "state_relative_l2_by_column": rel_cols.tolist(),
            "state_max_abs_by_column": np.max(abs(sc - sf), axis=0).tolist(),
            "omega_absolute_l2": float(np.sqrt(np.mean((oc - of) ** 2))),
            "omega_relative_l2": float(np.linalg.norm(oc - of) / np.linalg.norm(of)),
            "omega_max_abs": float(np.max(abs(oc - of)))}


def outside_energy_fraction(t, z, lo, hi):
    total = np.trapezoid(abs(z) ** 2, t)
    # Add interpolated boundary values so excluded integrals include partial
    # cells rather than silently assigning them to either side.
    def segment(a, b):
        if b <= a:
            return 0.0
        inside = (t >= a) & (t <= b)
        ts = np.concatenate(([a], t[inside], [b]))
        zs = np.interp(ts, t, z.real) + 1j * np.interp(ts, t, z.imag)
        return float(np.trapezoid(abs(zs) ** 2, ts))
    excluded = segment(t[0], min(lo, t[-1])) + segment(max(hi, t[0]), t[-1])
    return float(excluded / total), float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", type=Path, required=True)
    ap.add_argument("--fine", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    c, f = payload(args.coarse), payload(args.fine)
    assert np.all(np.diff(c["time"]) > 0) and np.all(np.diff(f["time"]) > 0)
    mode_metrics = {}
    supports = {}
    for origin in ("initial", "orbital_peak"):
        tc, zc0 = relative_mode(c, origin)
        tf, zf0 = relative_mode(f, origin)
        lo, hi = max(tc[0], tf[0]), min(tc[-1], tf[-1])
        dt = min(np.median(np.diff(tc)), np.median(np.diff(tf)))
        q = np.arange(lo, hi + dt * .25, dt); q = q[q <= hi]
        assert len(q) and q[0] >= lo and q[-1] <= hi
        metrics = {}
        for kind in ("linear", "cubic", "quintic"):
            zc = interpolate(tc, zc0, q, kind)
            zf = interpolate(tf, zf0, q, kind)
            assert np.isfinite(zc).all() and np.isfinite(zf).all()
            phase_residual = np.angle(zc * np.conj(zf))
            tail_c, total_c = outside_energy_fraction(tc, zc0, lo, hi)
            tail_f, total_f = outside_energy_fraction(tf, zf0, lo, hi)
            metrics[kind] = {
                "grid_dt_s": float(dt), "samples": int(len(q)),
                "relative_l2": float(np.linalg.norm(zc - zf) / np.linalg.norm(zf)),
                "phase_rms_rad": float(np.sqrt(np.mean(phase_residual ** 2))),
                "phase_max_abs_rad": float(max(abs(phase_residual))),
                "amplitude_relative_l2": float(np.linalg.norm(abs(zc) - abs(zf)) / np.linalg.norm(abs(zf))),
                "unmatched_support_energy_fraction": {"coarse": tail_c, "fine": tail_f},
                "integrated_total_energy": {"coarse": total_c, "fine": total_f},
            }
        mode_metrics[origin] = metrics
        supports[origin] = [float(lo), float(hi)]
    # Manufactured smooth chirp: quantify interpolation error independently
    # of waveform differences, on the same coarse/fine spacings.
    x = np.linspace(-1, 1, 2001)
    truth = np.exp(1j * (8 * x + 2 * x**2)) * (1 + .2 * x)
    xc, xf = x[::20], x[::5]
    control = {}
    for kind in ("linear", "cubic", "quintic"):
        yc = interpolate(xc, truth[::20], xf, kind)
        control[kind] = float(np.linalg.norm(yc - truth[::5]) / np.linalg.norm(truth[::5]))
    out = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(), "python": platform.python_version(),
        "torch": torch.__version__, "script_sha256": sha256(Path(__file__)),
        "coarse": c, "fine": f, "common_support_s": supports,
        "comparison_grid": "uniform at finer saved sampling interval; no extrapolation",
        "interpolation_metrics": mode_metrics,
        "trajectory_metrics": {origin: trajectory_metrics(c, f, origin)
                                for origin in ("initial", "orbital_peak")},
        "event_shifts_M": {"peak_fine_minus_coarse": f["peak_time_M"] - c["peak_time_M"],
                           "match_fine_minus_coarse": f["match_time_M"] - c["match_time_M"]},
        "manufactured_chirp_relative_l2": control,
        "manufactured_control_note": "analytic chirp sampled on fixed grids with coarse nodes every 20th point and fine nodes every 5th point; interpolation-only control",
        "finite_inputs_asserted": True,
    }
    # Avoid serializing arrays while retaining all scalar provenance.
    for item in (out["coarse"], out["fine"]):
        for key in ("time", "mode", "trajectory_time", "state", "omega", "valid"):
            item.pop(key, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
