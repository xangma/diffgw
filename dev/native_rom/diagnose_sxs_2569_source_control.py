"""Post-hoc pinned-source SEOBNRv4 control for the failed SXS:BBH:2569 row."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile

import lal
import lalsimulation as lalsim
import numpy as np
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC

from . import compare_sxs_2123 as c
from . import compare_sxs_2123_public_control as public
from . import measure_sxs_2508_weighted as weighted
from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_wheel_mode import _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/sxs_2569_source_control_plan.json"
SIMULATION = "SXS:BBH:2569"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--decoded-dir", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / record["path"] for name, record in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2569-source-control-plan.v1"
            or plan["simulation"] != SIMULATION
            or plan["masses_msun"] != [10, 22, 50, 100, 200]
            or plan["step_M"] != 0.25 or plan["source_capacity"] != 200000
            or plan["release_qualified"] is not False
            or any(q.sha256(path) != plan["inputs"][name]["sha256"]
                   for name, path in paths.items())):
        parser.error("frozen source-control inputs changed")
    cases = json.loads(paths["case_plan"].read_text())
    weighted_plan = json.loads(paths["weighted_plan"].read_text())
    comparison = json.loads(paths["comparison"].read_text())
    prior = json.loads(paths["public_control"].read_text())
    public_plan = json.loads(paths["public_plan"].read_text())
    source_plan = json.loads(paths["source_plan"].read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if (cases["schema"] != "diffgw.seobnrv4.sxs-third-holdout-plan.v1"
            or weighted_plan["schema"] != "diffgw.seobnrv4.sxs-third-holdout-weighted-plan.v1"
            or public_plan["schema"] != "diffgw.seobnrv4.sxs-2569-public-control-plan.v1"
            or prior["schema"] != "diffgw.seobnrv4.sxs-2569-public-control.v1"
            or prior["control_plan_sha256"] != q.sha256(paths["public_plan"])
            or prior["case_plan_sha256"] != q.sha256(paths["case_plan"])
            or prior["weighted_plan_sha256"] != q.sha256(paths["weighted_plan"])
            or prior["comparison_sha256"] != q.sha256(paths["comparison"])
            or prior["extraction_sha256"] != q.sha256(extraction_path)
            or comparison["extraction_sha256"] != q.sha256(extraction_path)
            or extraction["plan_sha256"] != q.sha256(paths["case_plan"])
            or source_plan["schema"] != "diffgw.seobnrv4.source-flux-initial-plan.v6"
            or prior["rows"][1]["total_mass_msun"] != 22
            or prior["rows"][1]["results"]["public_vs_nr"]["overlap"] >= 0.99):
        parser.error("source-control prerequisites do not match the failed case")
    q.lal_binary_identity(public_plan["binary_versions"]["lal"],
                          public_plan["binary_versions"]["lalsimulation"])
    libraries = _libraries(source_plan["pinned_wheel"])
    case = cases["cases"][SIMULATION]
    entry = comparison["cases"][SIMULATION]
    row, dt = case["row"], plan["step_M"]
    high = max(case["levels"])
    metadata_path = (args.archive_dir / SIMULATION.replace(":", "_")
                     / f"Lev{high}" / "metadata.json")
    metadata_record = extraction["cases"][SIMULATION]["archive_files"][
        f"Lev{high}:metadata.json"]
    if q.sha256(metadata_path) != metadata_record["sha256"]:
        parser.error("NR metadata changed")
    metadata = json.loads(metadata_path.read_text())
    if metadata["reference_time"] != metadata["relaxation_time"]:
        parser.error("NR reference time changed")
    orbital = float(np.linalg.norm(metadata["reference_orbital_frequency"]))
    if not math.isclose(orbital, float(np.linalg.norm(
            case["reference_orbital_frequency"])), abs_tol=1e-12):
        parser.error("NR orbital frequency changed")
    array_path = Path(entry["anchored_array"]["path"])
    if q.sha256(array_path) != entry["anchored_array"]["sha256"]:
        parser.error("anchored holdout array changed")
    nr_record = extraction["cases"][SIMULATION]["extracted"][f"Lev{high}"]
    nr_path = args.decoded_dir / nr_record["path"]
    if q.sha256(nr_path) != nr_record["sha256"]:
        parser.error("NR mode changed")
    report = dict(schema="diffgw.seobnrv4.sxs-2569-source-control.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  posthoc_selected_case=True, dominant_mode_proxy=True,
                  detector_noise_weighted=True, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  extraction_sha256=q.sha256(extraction_path),
                  nr_mode_sha256=q.sha256(nr_path),
                  nr_metadata_sha256=q.sha256(metadata_path),
                  anchored_array_sha256=q.sha256(array_path),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-sxs-2569-source-") as temp:
            private = Path(temp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(temp) / "source_mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_mode
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            print("compiled pinned source; generating exact holdout row", flush=True)
            epoch, delta_t, raw = _source(function, row, dt,
                                          plan["source_capacity"])
        mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
        if (not np.isfinite(raw).all()
                or abs(delta_t/mass_seconds-dt) > 1e-10):
            raise ValueError("invalid pinned-source waveform")
        scale = mass_seconds*float(C_SI)/(100*float(MPC_SI))
        source = c.anchor(dict(
            time=epoch/mass_seconds + np.arange(raw.size)*delta_t/mass_seconds,
            mode=raw/scale), dt=dt)
        wheel = c.anchor(public.public_mode(row, dt, public_plan["public_call"]),
                         dt=dt)
        nr_high = c.anchor(c.load_mode(nr_path), dt=dt)
        with np.load(array_path, allow_pickle=False) as data:
            waves = {name: dict(time=data[f"{name}_time"],
                                mode=data[f"{name}_mode"])
                     for name in ("native", "nr_low", "nr_high")}
        waves["source"] = source
        waves["public"] = wheel
        settled_start = (metadata["relaxation_time"]
                         -entry["anchor"][f"Lev{high}"]["anchor_time_M"])
        frequency, spectra, support, fft_length = weighted.tapered_spectra(
            waves, lower=settled_start, dt=dt,
            width=weighted_plan["taper_width_M"])
        windows = cases["method"][
            "windows_relative_to_highest_resolution_amplitude_peak_M"]
        fixed_support = entry["comparisons"]["native_vs_high"]["common_support_M"]
        report["fixed_crossing"] = dict(
            source_vs_nr=c.compare(source, nr_high, windows, dt,
                                   support=fixed_support),
            source_vs_public=c.compare(source, wheel, windows, dt,
                                       support=fixed_support))
        report["anchors"] = {name: {key: value for key, value in wave.items()
                                   if key not in ("time", "mode")}
                             for name, wave in (("source", source), ("public", wheel))}
        report["source_mode_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
        report["source_samples"] = raw.size
        report["source_epoch_s"] = epoch
        report["source_delta_t_s"] = delta_t
        report["common_time_support_M"] = support
        report["prior_public_support_M"] = prior["common_time_support_M"]
        report["fft_length"] = fft_length
        for mass in plan["masses_msun"]:
            seconds = mass*float(MSUN_SEC)
            frequency_hz = frequency/seconds
            lower_hz = max(weighted_plan["minimum_frequency_Hz"],
                           orbital/(math.pi*seconds))
            band = ((frequency_hz >= lower_hz)
                    & (frequency_hz <= weighted_plan["maximum_frequency_Hz"]))
            psd = np.array([
                lalsim.SimNoisePSDaLIGOZeroDetHighPower(float(value))
                for value in frequency_hz[band]])
            results = {name: weighted.optimized_overlap(
                frequency[band], spectra[candidate][band],
                spectra[reference][band], psd,
                bounds=tuple(weighted_plan["allowed_time_shift_M"]),
                step=weighted_plan["time_shift_grid_step_M"])
                for name, candidate, reference in (
                    ("source_vs_nr", "source", "nr_high"),
                    ("public_vs_nr", "public", "nr_high"),
                    ("native_vs_nr", "native", "nr_high"),
                    ("low_vs_high_nr", "nr_low", "nr_high"),
                    ("source_vs_public", "source", "public"))}
            report["rows"].append(dict(total_mass_msun=mass,
                                       lower_frequency_Hz=lower_hz,
                                       frequency_bins=int(np.count_nonzero(band)),
                                       results=results))
            q.write_evidence(args.json_out, report)
            print(f"mass={mass} source/NR={results['source_vs_nr']['overlap']:.9f}",
                  flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
