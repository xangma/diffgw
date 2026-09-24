"""Post-hoc, unweighted time/phase alignment control for SXS:BBH:2123."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as pc
from . import qualify_generator as q


def optimized_pair(candidate: dict, reference: dict, *, window: tuple[float, float],
                   dt: float, lag_bounds: tuple[float, float]) -> dict:
    peak = reference["peak_after_anchor_M"]
    grid = np.arange(math.ceil((peak+window[0])/dt),
                     math.floor((peak+window[1])/dt)+1)*dt
    if len(grid) < 100 or grid[0] < reference["time"][0] or grid[-1] > reference["time"][-1]:
        raise ValueError("reference does not cover frozen comparison window")
    if (grid[0]+lag_bounds[0] < candidate["time"][0]
            or grid[-1]+lag_bounds[1] > candidate["time"][-1]):
        raise ValueError("candidate does not cover every allowed time shift")
    def sample(waveform, times):
        mode = waveform["mode"]
        return np.interp(times, waveform["time"], mode.real) + 1j*np.interp(
            times, waveform["time"], mode.imag)
    expected = sample(reference, grid)
    expected_norm = np.linalg.norm(expected)
    def overlap(lag):
        actual = sample(candidate, grid+lag)
        return abs(np.vdot(actual, expected))/(np.linalg.norm(actual)*expected_norm)
    lags = np.arange(lag_bounds[0], lag_bounds[1]+dt/2, dt)
    scores = np.array([overlap(lag) for lag in lags])
    best = int(np.argmax(scores))
    lower = max(lag_bounds[0], float(lags[best]-dt))
    upper = min(lag_bounds[1], float(lags[best]+dt))
    if lower == upper:
        lag = lower
    else:
        optimum = minimize_scalar(lambda x: -overlap(x), bounds=(lower, upper),
                                  method="bounded", options={"xatol": 1e-7})
        lag = float(optimum.x)
    actual = sample(candidate, grid+lag)
    phase = float(np.angle(np.vdot(actual, expected)))
    rotated = actual*np.exp(1j*phase)
    return dict(window_relative_to_reference_peak_M=list(window), samples=len(grid),
                fixed_anchor_overlap=float(overlap(0.)),
                optimized_overlap=float(overlap(lag)),
                best_lag_M=lag, best_phase_rad=phase,
                time_phase_adjusted_metrics=q.mode_metrics(rotated, expected),
                grid_search_max_overlap=float(scores[best]),
                time_phase_optimized=True, amplitude_optimized=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--optimized-plan", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--public-control", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    optimized = json.loads(args.optimized_plan.read_text())
    if (optimized["schema"] != "diffgw.seobnrv4.sxs-2123-optimized-plan.v1"
            or optimized["primary_comparison_sha256"] != q.sha256(args.prior)
            or optimized["public_control_sha256"] != q.sha256(args.public_control)):
        raise ValueError("optimized diagnostic does not match frozen prerequisites")
    plan = json.loads(args.plan.read_text())
    public_control = json.loads(args.public_control.read_text())
    public_plan_path = args.public_control.parent / "sxs_2123_public_control_plan.json"
    if public_control["control_plan_sha256"] != q.sha256(public_plan_path):
        raise ValueError("public control plan changed")
    public_plan = json.loads(public_plan_path.read_text())
    prior = json.loads(args.prior.read_text())
    if prior["plan_sha256"] != q.sha256(args.plan):
        raise ValueError("primary comparison plan changed")
    extraction_path = args.decoded_dir / "extraction.json"
    if prior["extraction_sha256"] != q.sha256(extraction_path):
        raise ValueError("decoded NR manifest changed")
    extracted = json.loads(extraction_path.read_text())
    nr = {}
    for lev in (3, 4):
        record = extracted["extracted"][f"Lev{lev}"]
        path = args.decoded_dir / record["path"]
        if q.sha256(path) != record["sha256"]:
            raise ValueError(f"decoded Lev{lev} changed")
        nr[f"Lev{lev}"] = c.anchor(c.load_mode(path))
    import torch
    torch.set_num_threads(1)
    method = plan["method"]
    native = q.native_run(plan["row"], dict(
        name="native", atol=method["native_atol"], rtol=method["native_rtol"],
        dt_geom=method["grid_step_M"]), q.Settings(max_run_seconds=600.))
    if q.summarize_run(native)["array_sha256"] != prior["native"]["native"]["array_sha256"]:
        raise ValueError("native mode changed")
    waves = dict(native=c.anchor(native), public=c.anchor(pc.public_mode(
        plan["row"], method["grid_step_M"], public_plan["public_call"])), **nr)
    window = tuple(optimized["window_relative_to_Lev4_amplitude_peak_M"])
    bounds = tuple(optimized["allowed_time_shift_M"])
    dt = optimized["grid_step_M"]
    pairs = dict(native_vs_Lev4=optimized_pair(waves["native"], waves["Lev4"],
                                              window=window, dt=dt, lag_bounds=bounds),
                 public_vs_Lev4=optimized_pair(waves["public"], waves["Lev4"],
                                              window=window, dt=dt, lag_bounds=bounds),
                 Lev3_vs_Lev4=optimized_pair(waves["Lev3"], waves["Lev4"],
                                            window=window, dt=dt, lag_bounds=bounds))
    result = dict(schema="diffgw.seobnrv4.sxs-2123-optimized-diagnostic.v1",
                  plan_sha256=q.sha256(args.plan),
                  optimized_plan_sha256=q.sha256(args.optimized_plan),
                  primary_comparison_sha256=q.sha256(args.prior),
                  public_control_sha256=q.sha256(args.public_control),
                  script_sha256=q.sha256(__file__), pairs=pairs,
                  detector_noise_weighted=False, release_qualified=False,
                  interpretation="This is a post-hoc, unweighted single-row alignment diagnostic. "
                                 "It is not the published detector-weighted faithfulness or a "
                                 "physical-validation gate.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: pair["optimized_overlap"] for name, pair in pairs.items()}))


if __name__ == "__main__":
    main()
