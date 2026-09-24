"""Post-hoc public SEOBNRv4 control for the frozen SXS:BBH:2123 comparison.

This diagnoses whether a native-versus-NR residual is also present in the
pinned public model. It is not an independent holdout or release criterion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as c
from . import qualify_generator as q


def public_mode(row: dict, dt_geom: float, public_call: dict) -> dict:
    import lal
    import lalsimulation as lalsim
    from scripts.verify_eob_published_waveforms import _invert_22
    from torchwave.constants import C_SI, MPC_SI, MSUN_SEC

    mass_seconds = (row["mass1"]+row["mass2"])*float(MSUN_SEC)
    delta_t = dt_geom*mass_seconds
    f_lower = row["omega_start"]/(math.pi*mass_seconds)
    distance = public_call["distance_Mpc"]
    inclination = public_call["inclination_rad"]
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        row["mass1"]*lal.MSUN_SI, row["mass2"]*lal.MSUN_SI,
        0., 0., row["chi1"], 0., 0., row["chi2"],
        distance*1e6*lal.PC_SI, inclination, public_call["phase_rad"], 0., 0., 0.,
        delta_t, f_lower, public_call["f_ref_Hz"], lal.CreateDict(),
        lalsim.GetApproximantFromString(public_call["approximant"]))
    if (len(hp.data.data) != len(hc.data.data) or hp.epoch != hc.epoch
            or hp.deltaT != hc.deltaT):
        raise ValueError("inconsistent public polarization grids")
    time = float(hp.epoch)/mass_seconds + np.arange(len(hp.data.data))*float(hp.deltaT)/mass_seconds
    mode = _invert_22(np.asarray(hp.data.data), np.asarray(hc.data.data), inclination)
    mode /= mass_seconds*float(C_SI)/(distance*float(MPC_SI))
    return dict(time=time, mode=mode, f_lower_Hz=f_lower,
                delta_t_s=delta_t, distance_Mpc=distance, inclination_rad=inclination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--control-plan", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    control = json.loads(args.control_plan.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2123-plan.v1"
            or control["schema"] != "diffgw.seobnrv4.sxs-2123-public-control-plan.v1"
            or control["primary_comparison_sha256"] != q.sha256(args.prior)):
        raise ValueError("public control does not match frozen primary result")
    prior = json.loads(args.prior.read_text())
    if prior["plan_sha256"] != q.sha256(args.plan):
        raise ValueError("primary comparison does not match SXS plan")
    extraction_path = args.decoded_dir / "extraction.json"
    if q.sha256(extraction_path) != prior["extraction_sha256"]:
        raise ValueError("decoded SXS input changed")
    extracted = json.loads(extraction_path.read_text())
    nr_path = args.decoded_dir / extracted["extracted"]["Lev4"]["path"]
    if q.sha256(nr_path) != extracted["extracted"]["Lev4"]["sha256"]:
        raise ValueError("decoded Lev4 mode changed")
    identity = q.lal_binary_identity(control["binary_versions"]["lal"],
                                     control["binary_versions"]["lalsimulation"])
    method = plan["method"]
    import torch
    torch.set_num_threads(1)
    native = q.native_run(plan["row"], dict(
        name="native", atol=method["native_atol"], rtol=method["native_rtol"],
        dt_geom=method["grid_step_M"]), q.Settings(max_run_seconds=600.))
    if (q.summarize_run(native)["array_sha256"]
            != prior["native"]["native"]["array_sha256"]):
        raise ValueError("native waveform changed after primary comparison")
    public = public_mode(plan["row"], method["grid_step_M"], control["public_call"])
    a_public = c.anchor(public)
    a_native = c.anchor(native)
    a_nr = c.anchor(c.load_mode(nr_path))
    windows = method["windows_relative_to_Lev4_amplitude_peak_M"]
    dt = method["grid_step_M"]
    result = dict(schema="diffgw.seobnrv4.sxs-2123-public-control.v1",
                  plan_sha256=q.sha256(args.plan),
                  control_plan_sha256=q.sha256(args.control_plan),
                  primary_comparison_sha256=q.sha256(args.prior),
                  script_sha256=q.sha256(__file__), lal_binary=identity,
                  public_call={key: public[key] for key in (
                      "f_lower_Hz", "delta_t_s", "distance_Mpc", "inclination_rad")},
                  anchors={name: {key: value for key, value in wave.items()
                                 if key not in ("time", "mode")}
                           for name, wave in (("public", a_public), ("native", a_native),
                                              ("Lev4", a_nr))},
                  public_vs_Lev4=c.compare(a_public, a_nr, windows, dt),
                  native_vs_public=c.compare(a_native, a_public, windows, dt),
                  release_qualified=False,
                  interpretation="Post-hoc diagnostic: a shared discrepancy with NR cannot be "
                                 "assigned to native implementation alone. No time/phase fit or "
                                 "detector-weighted faithfulness is performed.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(public_vs_Lev4=result["public_vs_Lev4"]["metrics"]
                          ["complex_relative_l2"],
                          native_vs_public=result["native_vs_public"]["metrics"]
                          ["complex_relative_l2"])))


if __name__ == "__main__":
    main()
