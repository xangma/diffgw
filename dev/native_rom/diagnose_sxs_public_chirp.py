"""Check whether the failed NR chirp is shared by pinned public SEOBNRv4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as compare
from . import compare_sxs_2123_public_control as public
from . import diagnose_sxs_cross_run_chirp as chirp
from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/sxs_public_chirp_plan.json"
OUTPUT = q.ROOT / "dev/native_rom/evidence/2026-09-25/sxs_public_chirp.json"


def _read_checked(entry: dict) -> dict:
    path = Path(entry["path"])
    if q.sha256(path) != entry["sha256"]:
        raise ValueError(f"frozen input changed: {path}")
    return json.loads(path.read_text())


def main() -> None:
    if OUTPUT.exists():
        raise ValueError("output already exists")
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-public-chirp-plan.v1"
            or plan["release_qualified"] is not False
            or plan["posthoc"] is not True
            or plan["cases"] != {"third": "SXS:BBH:2569", "paper": "SXS:BBH:0260"}):
        raise ValueError("frozen public chirp plan changed")
    earlier = _read_checked(plan["prior_chirp_result"])
    chirp_plan = _read_checked(plan["prior_chirp_plan"])
    if (earlier["status"] != "diagnostic_complete"
            or earlier["plan_sha256"] != plan["prior_chirp_plan"]["sha256"]
            or chirp_plan["sample_step_M"] != 0.25):
        raise ValueError("earlier fixed-frequency comparison changed")
    cases = {}
    for label, simulation in plan["cases"].items():
        inputs = plan["inputs"][label]
        case_plan = _read_checked(inputs["case_plan"])
        parity_plan = _read_checked(inputs["parity_plan"])
        parity = _read_checked(inputs["parity_result"])
        if (parity["plan_sha256"] != inputs["parity_plan"]["sha256"]
                or parity_plan["case_plan_sha256"] != inputs["case_plan"]["sha256"]
                or parity_plan["public_call"]["approximant"] != "SEOBNRv4"
                or parity_plan["binary_versions"] != {"lal": "7.7.1",
                                                      "lalsimulation": "6.2.1"}):
            raise ValueError(f"pinned public parity changed for {simulation}")
        row = case_plan["cases"][simulation]["row"]
        generated = compare.anchor(public.public_mode(
            row, chirp_plan["sample_step_M"], parity_plan["public_call"]),
            dt=chirp_plan["sample_step_M"])
        prior_anchor = parity["cases"][simulation]["public_anchor"]
        for key in ("anchor_time_M", "peak_after_anchor_M", "peak_amplitude"):
            if abs(generated[key] - prior_anchor[key]) > 1e-8:
                raise ValueError(f"pinned public wave changed for {simulation}: {key}")
        entry = chirp_plan["arrays"][label]
        path = Path(entry["path"])
        if q.sha256(path) != entry["sha256"]:
            raise ValueError(f"frozen {simulation} native/NR array changed")
        with np.load(path, allow_pickle=False) as source:
            cases[label] = dict(public=generated,
                                native=dict(time=source["native_time"],
                                            mode=source["native_mode"]),
                                high=dict(time=source["nr_high_time"],
                                          mode=source["nr_high_mode"]))
    result = dict(schema="diffgw.seobnrv4.sxs-public-chirp.v1", status="running",
                  release_qualified=False, posthoc=True,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  lal_binary=q.lal_binary_identity("7.7.1", "6.2.1"),
                  mode_angular_frequencies_M_inv=chirp_plan[
                      "mode_angular_frequencies_M_inv"], cases=plan["cases"],
                  smoothing={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    anchor_index = chirp_plan["mode_angular_frequencies_M_inv"].index(
        chirp_plan["phase_anchor_mode_frequency_M_inv"])
    for width in chirp_plan["smoothing_widths_M"]:
        entries = {}
        for label, waves in cases.items():
            crossed = {name: chirp._crossings(wave, chirp_plan, width)
                       for name, wave in waves.items()}
            native_nr = chirp._residual(crossed["native"], crossed["high"],
                                        anchor_index)
            old_native_nr = earlier["smoothing"][str(width)][label][
                "native_vs_high_nr"]
            if any(np.max(abs(np.array(native_nr[key]) - old_native_nr[key])) > 1e-10
                   for key in native_nr):
                raise ValueError("native/NR chirp control changed")
            entries[label] = dict(
                public_vs_high_nr=chirp._residual(crossed["public"], crossed["high"],
                                                  anchor_index),
                public_vs_native=chirp._residual(crossed["public"], crossed["native"],
                                                 anchor_index))
        result["smoothing"][str(width)] = entries
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    for label in plan["cases"]:
        row = result["smoothing"]["20"][label]
        print(label, row["public_vs_high_nr"]["phase_difference_from_anchor_rad"][0],
              row["public_vs_native"]["phase_difference_from_anchor_rad"][0],
              flush=True)


if __name__ == "__main__":
    main()
