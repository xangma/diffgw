"""Dev-only q4 source/wheel/native ringdown-stage attribution."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, fitted_peak, public_mode
from dev.native_rom.diagnose_q4_public_ringdown_inverse import _fit, _predictions
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_nqc_inputs import INCLUDE
from dev.native_rom.diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from dev.native_rom.diagnose_source_q4_ultrafine_chart import _native
from dev.native_rom.diagnose_source_wheel_mode import _public, _source
from torchwave.approximants.eob.merger_ringdown import (
    seobnrv4_merger_shape,
    seobnrv4_peak_preserving_shape,
)
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "q4_ultrafine_ringdown_stage_plan.json"
OUTPUT = EVIDENCE / "q4_ultrafine_ringdown_stage.json"


def _wheel_qnm(row):
    ratio = row["mass1"] / row["mass2"]
    status, final_mass, final_spin = lalsim.SimIMREOBFinalMassSpin(
        ratio, 1., [0., 0., row["chi1"]], [0., 0., row["chi2"]],
        lalsim.SEOBNRv4,
    )
    if status:
        raise RuntimeError("pinned-wheel remnant fit failed")
    vector = lal.CreateCOMPLEX16Vector(1)
    if lalsim.SimIMREOBGenerateQNMFreqV2fromFinal(
            vector, 1., final_spin, 2, 2, 1):
        raise RuntimeError("pinned-wheel QNM lookup failed")
    sigma = complex(vector.data[0]) * lal.MTSUN_SI / final_mass
    return dict(frequency_M=float(sigma.real), damping_M=float(sigma.imag),
                final_mass_fraction=float(final_mass), final_spin=float(final_spin))


def _shape(row, qnm):
    arguments = [torch.tensor(row[key], dtype=torch.float64)
                 for key in ("mass1", "mass2", "chi1", "chi2")]
    nu, chi = seobnrv4_nqc_parameters(*arguments)
    sigma = torch.tensor(complex(qnm["frequency_M"], -qnm["damping_M"]),
                         dtype=torch.complex128)
    return [float(x) for x in seobnrv4_peak_preserving_shape(
        sigma, seobnrv4_merger_shape(nu, chi))]


def _fit_mode(chart, mass_seconds, qnm, shape, plan):
    time, mode = chart
    peak = fitted_peak(time, mode, "quadratic_power")
    time_M = (time - peak["time_s"]) / mass_seconds
    amplitude = _fit(time_M, mode, shape, qnm["frequency_M"],
                     qnm["damping_M"], plan["fit_window_M"],
                     plan["origin_bounds_M"], "amplitude")
    phase = _fit(time_M, mode, shape, qnm["frequency_M"],
                 qnm["damping_M"], plan["fit_window_M"],
                 plan["origin_bounds_M"], "phase")
    holdout = []
    for lower, upper in plan["holdout_windows_M"]:
        selected = (time_M >= lower) & (time_M <= upper)
        if selected.sum() < 20:
            raise ValueError("too few ringdown holdout samples")
        predicted_amplitude, predicted_phase = _predictions(
            time_M[selected], amplitude, phase, shape,
            qnm["frequency_M"], qnm["damping_M"])
        observed_phase = -np.unwrap(np.angle(mode[selected]))
        phase_error = np.angle(np.exp(1j * (observed_phase - predicted_phase)))
        holdout.append(dict(window_M=[lower, upper], samples=int(selected.sum()),
                            log_amplitude_rms=float(np.sqrt(np.mean((
                                np.log(abs(mode[selected])) - np.log(predicted_amplitude))**2))),
                            phase_rms_rad=float(np.sqrt(np.mean(phase_error**2)))))
    return dict(mode_sha256=hashlib.sha256(mode.tobytes()).hexdigest(),
                peak=peak, qnm=qnm, shape=shape, amplitude=amplitude, phase=phase,
                holdout=holdout)


def _normalized(result, elapsed):
    amplitude = result["amplitude"]
    phase = result["phase"]
    qnm = result["qnm"]
    shape = result["shape"]
    # Separate amplitude and phase inverse fits may infer slightly different
    # origins; evaluate each at its own fitted boundary.
    a, _ = _predictions(elapsed + amplitude["t0_M"], amplitude, phase,
                        shape, qnm["frequency_M"], qnm["damping_M"])
    _, p = _predictions(elapsed + phase["t0_M"], amplitude, phase,
                        shape, qnm["frequency_M"], qnm["damping_M"])
    return a / amplitude["amplitude"] * np.exp(-1j * (p - phase["phase"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    paths = {
        "fixed_frequency": EVIDENCE / "q4_ultrafine_fixed_frequency.json",
        "nqc_inputs": EVIDENCE / "source_q4_ultrafine_nqc_inputs.json",
        "nqc_override": EVIDENCE / "source_q4_ultrafine_nqc_override.json",
        "ultrafine_chart": EVIDENCE / "source_q4_ultrafine_chart.json",
        "source_plan": ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json",
        "adapter": ROOT / "dev/native_rom/reference/full_v4_source_large_nqc_override_adapter.c",
    }
    if (plan["schema"] != "diffgw.seobnrv4.q4-ultrafine-ringdown-stage-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_step_M"] != 0.015625
            or plan["fit_window_M"] != [0.5, 30.]
            or plan["holdout_windows_M"] != [[30., 50.], [50., 80.]]
            or plan["origin_bounds_M"] != [-0.5, 0.5]
            or plan["normalized_elapsed_M"] != [0.5, 80.]
            or plan["capacity"] != 50000
            or plan["atol"] != 1e-14 or plan["rtol"] != 1e-13
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        raise ValueError("frozen q4 ringdown-stage plan changed")
    prior = json.loads(paths["fixed_frequency"].read_text())
    inputs = json.loads(paths["nqc_inputs"].read_text())
    override = json.loads(paths["nqc_override"].read_text())
    chart_evidence = json.loads(paths["ultrafine_chart"].read_text())
    source_plan = json.loads(paths["source_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or inputs["status"] != "diagnostic_complete"
            or override["status"] != "diagnostic_complete"
            or chart_evidence["status"] != "diagnostic_complete"
            or any(item["row"] != prior["row"]
                   for item in (inputs, override, chart_evidence))):
        raise ValueError("frozen q4 source or native evidence changed")
    row = prior["row"]
    step = plan["output_step_M"]
    key = str(step)
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    historical = json.loads((ROOT / "dev/native_rom/evidence/2026-09-24/"
                             "combined_domain_numerics.json").read_text())
    settings = q.Settings(**historical["settings"])
    torch.set_num_threads(1)
    result = dict(schema="diffgw.seobnrv4.q4-ultrafine-ringdown-stage.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, modes={}, comparisons={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="diffgw-q4-ringdown-stage-") as temp:
        private = Path(temp) / "source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
        original = source_file.read_text()
        if original.count(INCLUDE) != 1:
            raise ValueError("pinned source NQC insertion point changed")
        source_file.write_text(original.replace(INCLUDE, INCLUDE + OVERRIDE))
        library_path = Path(temp) / "source_override.dylib"
        _compile(private, args.headers, args.gsl_include,
                 _libraries(source_plan["pinned_wheel"]), paths["adapter"], library_path,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"], require_wheel_flux=False)
        library = ctypes.CDLL(str(library_path))
        baseline = library.diffgw_v4_full_source_mode
        baseline.argtypes = [ctypes.c_double] * 6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        baseline.restype = ctypes.c_int
        modified = library.diffgw_v4_full_source_mode_with_nqc
        modified.argtypes = baseline.argtypes + [ctypes.POINTER(ctypes.c_double)]
        modified.restype = ctypes.c_int
        source = _source(baseline, row, step, plan["capacity"])
        source_wheel = override_source(modified, row, step, plan["capacity"],
            np.asarray(override["grids"][key]["wheel_active"]))
        wheel = _public(row, step)[:3]
        run = _native(row, step, plan, settings, return_boundary_data=True)
        native_time, native_mode, _, _, actual_mass_seconds = public_mode(row, run)
        if abs(actual_mass_seconds - mass_seconds) > 1e-15:
            raise ValueError("native mass conversion changed")
        charts = {
            "source": (source[0] + np.arange(len(source[2])) * source[1], source[2]),
            "source_wheel_nqc": (source_wheel[0] + np.arange(len(source_wheel[2]))
                                 * source_wheel[1], source_wheel[2]),
            "wheel": (wheel[0] + np.arange(len(wheel[2])) * wheel[1], wheel[2]),
            "native": (native_time, native_mode),
        }
        expected = prior["mode_sha256"]
        if any(hashlib.sha256(charts[name][1].tobytes()).hexdigest()
               != expected["source_wheel_coefficients" if name == "source_wheel_nqc"
                           else name] for name in charts):
            raise ValueError("frozen q4 whole-waveform arrays changed")
        if (q.summarize_run(run)["array_sha256"]
                != chart_evidence["grids"][key]["native"]["array_sha256"]):
            raise ValueError("frozen native mode changed")
        wheel_qnm = _wheel_qnm(row)
        boundary = run["boundary_data"]
        native_qnm = dict(frequency_M=boundary["qnm_frequency"],
                          damping_M=boundary["qnm_damping"],
                          final_mass_fraction=run["final_mass_fraction"],
                          final_spin=run["final_spin"])
        for name, chart in charts.items():
            qnm = native_qnm if name == "native" else wheel_qnm
            result["modes"][name] = _fit_mode(
                chart, mass_seconds, qnm, _shape(row, qnm), plan)
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        match = inputs["grids"][key]["match_time_M"]
        source_boundary_from_orbital_peak = np.floor(match / step) * step - inputs[
            "grids"][key]["orbital_peak_M"]
        native_boundary_from_orbital_peak = run["boundary_time_M"] - run["peak_time_M"]
        for name in ("source", "source_wheel_nqc", "native"):
            if name == "native":
                offset = native_boundary_from_orbital_peak
            else:
                offset = source_boundary_from_orbital_peak
            mode_peak_from_orbital_peak = result["modes"][name]["peak"]["time_s"] / mass_seconds
            # The public/source epoch is already orbital-peak relative; the
            # native time returned by public_mode is likewise peak relative.
            result["modes"][name]["expected_boundary_from_mode_peak_M"] = (
                offset - mode_peak_from_orbital_peak)
            result["modes"][name]["fitted_minus_expected_boundary_M"] = (
                result["modes"][name]["amplitude"]["t0_M"]
                - result["modes"][name]["expected_boundary_from_mode_peak_M"])
        elapsed = np.linspace(*plan["normalized_elapsed_M"], 801)
        normalized = {name: _normalized(item, elapsed)
                      for name, item in result["modes"].items()}
        for candidate, reference in (("native", "source"), ("native", "wheel"),
                                     ("source", "wheel"), ("source_wheel_nqc", "wheel")):
            left = result["modes"][candidate]
            right = result["modes"][reference]
            left_anchor = prior["anchors"]["source_wheel_coefficients"
                                           if candidate == "source_wheel_nqc" else candidate]
            right_anchor = prior["anchors"][reference]
            peak_aligned = fitted_metrics(*charts[candidate], *charts[reference],
                                          mass_seconds, "quadratic_power")
            result["comparisons"][candidate + "_vs_" + reference] = dict(
                normalized_shape=q.mode_metrics(normalized[candidate], normalized[reference]),
                peak_aligned_post_peak=peak_aligned["windows"]["post_peak"],
                peak_lag_from_frequency_anchor_M=(
                    left_anchor["time_before_peak_M"]
                    - right_anchor["time_before_peak_M"]),
                peak_phase_difference_from_frequency_anchor_rad=float(np.angle(np.exp(
                    1j * ((left_anchor["peak"]["phase_rad"] - left_anchor["phase_rad"])
                          - (right_anchor["peak"]["phase_rad"]
                             - right_anchor["phase_rad"]))))),
                fitted_boundary_time_difference_M=(left["amplitude"]["t0_M"]
                                                   - right["amplitude"]["t0_M"]),
                fitted_boundary_amplitude_relative_difference=(
                    left["amplitude"]["amplitude"] / right["amplitude"]["amplitude"] - 1),
                fitted_boundary_frequency_difference_M_inv=(
                    left["phase"]["frequency_M"] - right["phase"]["frequency_M"]),
                qnm_frequency_difference_M_inv=(left["qnm"]["frequency_M"]
                                                - right["qnm"]["frequency_M"]),
            )
    result["elapsed_s"] = time.monotonic() - started
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, comparison in result["comparisons"].items():
        print(name, comparison["normalized_shape"]["complex_relative_l2"], flush=True)


if __name__ == "__main__":
    main()
