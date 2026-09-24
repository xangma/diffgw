"""Compare SXS-row fine-stage samplers with a local DOP853 reference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_basis_trace import _basis, _local_fit, _physics
from .diagnose_boundary_strong_spin_ode import array, array_hash
from .diagnose_rkf45_offgrid_reference import _errors, _locally_seeded_reference
from .eob.torch.adaptive_driver_clean import (
    _diagnostic_hermite_output, _diagnostic_rkf45_output,
    reconstruct_seobnrv4_22_adaptive,
)
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sampling-result", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    previous = json.loads(args.sampling_result.read_text())
    reference_plan = json.loads(args.reference_plan.read_text())
    if (reference_plan["schema"] != "diffgw.seobnrv4.sxs-2123-sampling-reference-plan.v1"
            or reference_plan["prior_sampling_sha256"] != q.sha256(args.sampling_result)
            or previous["plan_sha256"] != q.sha256(args.plan)
            or previous["accepted_arrays_identical"] is not True
            or reference_plan["level_name"] != "tightest"
            or reference_plan["probe"] != "fine-stage local match time + 0.0078125 M + k*0.0625 M for k=-7,...,7"):
        raise ValueError("sampling reference does not match frozen inputs")
    torch.set_num_threads(1)
    level = previous["level"]
    started = time.monotonic()
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(plan["row"]["omega_start"], dtype=torch.float64),
            **{key: plan["row"][key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES, **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
            ringdown_duration=150.0, num_steps=20000, max_accepted_steps=20000,
            max_evaluations=200000, max_rejected_steps=10000)
    hashes = dict(mode=array_hash(array(result.mode)), time=array_hash(array(result.time)))
    if hashes != previous["runs"]["default_replay"]["native"]["array_sha256"]:
        raise ValueError("default waveform differs from frozen sampling comparison")
    fine = result.trajectory.fine
    valid = array(fine.valid).astype(bool)
    accepted_time = array(fine.time)[valid]
    accepted_state = array(fine.state)[valid]
    accepted_derivative = array(fine.derivative)[valid]
    field, _ = _physics(plan["row"])
    field_difference = []
    for index in (0, len(accepted_time)//2, len(accepted_time)-1):
        with torch.no_grad():
            value = array(field(torch.from_numpy(accepted_state[index])))
        field_difference.append(float(max(abs(value-accepted_derivative[index]))))
    if max(field_difference) > reference_plan["reference"]["vector_field_derivative_tolerance"]:
        raise ValueError("reference vector field differs from native accepted derivatives")
    match = float(result.attachment.match_time)-float(result.trajectory.restart_time)
    query = match + 0.0078125 + np.arange(-7, 8)*0.0625
    if query[0] <= accepted_time[0] or query[-1] >= accepted_time[-1]:
        raise ValueError("probe exceeds fine accepted history")
    probe = replace(fine, output_time=torch.from_numpy(query))
    with torch.no_grad():
        states = dict(
            natural_cubic=array(natural_cubic_spline(
                torch.from_numpy(accepted_time), torch.from_numpy(accepted_state),
                scalar_output=False).evaluate(torch.from_numpy(query))),
            hermite=array(_diagnostic_hermite_output(probe).output_state),
            rkf45=array(_diagnostic_rkf45_output(probe, field).output_state))
    references = []
    evaluations = []
    for rtol, atol in reference_plan["reference"]["rtol_atol"]:
        values, count = _locally_seeded_reference(
            field, accepted_time, accepted_state, query, rtol, atol)
        references.append(values)
        evaluations.append(count)
    tight = references[-1]
    window = reference_plan["reference"]["local_phase_second_window_M"]
    chart = (query-match)/window
    phase_second = {name: _local_fit(_basis(field, values), chart, window).tolist()
                    for name, values in (list(states.items()) +
                                         [("dop853_loose", references[0]),
                                          ("dop853_tight", tight)])}
    result_json = dict(schema="diffgw.seobnrv4.sxs-2123-sampling-reference.v1",
                       plan_sha256=q.sha256(args.plan),
                       prior_sampling_sha256=q.sha256(args.sampling_result),
                       reference_plan_sha256=q.sha256(args.reference_plan),
                       script_sha256=q.sha256(__file__),
                       generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                       row=plan["row"], level=level, replay_array_sha256=hashes,
                       elapsed_seconds=time.monotonic()-started,
                       local_match_time_M=match, probe_time_M=query.tolist(),
                       accepted_brackets=int(len(np.unique(
                           np.searchsorted(accepted_time, query, side="right")-1))),
                       field_derivative_max_abs=field_difference,
                       dop853_evaluations=evaluations,
                       state_error={**{name: _errors(value, tight)
                                      for name, value in states.items()},
                                    "reference_spread": _errors(references[0], tight)},
                       phase_basis_second=phase_second,
                       phase_basis_second_error={name: (np.asarray(value)-np.asarray(
                           phase_second["dop853_tight"])).tolist()
                                                 for name, value in phase_second.items()
                                                 if name != "dop853_tight"},
                       release_qualified=False,
                       interpretation="Local independent numerical sampling reference "
                                      "on the same ODE; no physical or ROM validation.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result_json, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(max_abs_state={name: data["max_abs_any"]
                                          for name, data in result_json["state_error"].items()},
                          field_derivative_max_abs=max(field_difference))), flush=True)


if __name__ == "__main__":
    main()
