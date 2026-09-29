"""Preregistered nonspinning chain-residual and grid-path controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.eob.torch import adaptive_driver_clean as driver
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline
from torchwave.approximants.eob.nqc import nqc_22_basis
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "ode_gate_slice" / "rerun_nonspin.json"
GATE = HERE.parent / "ode_chain_derivatives" / "gate_nonspin.json"


def floats(tensor):
    return tensor.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("levels", "control"), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    campaign = json.loads(CAMPAIGN.read_text())
    gate = json.loads(GATE.read_text())
    row = campaign["results"][0]["inputs"]
    assert row == dict(mass1=30.0, mass2=20.0, chi1=0.0, chi2=0.0,
                       omega_start=0.04)
    settings = q.Settings(**campaign["config"]["settings"])
    plans = {item["name"]: item for item in q.run_plan(settings)}
    prior = {item["name"]: item for item in gate["runs"]}
    for name in plans:
        assert all(plans[name][key] == prior[name][key]
                   for key in ("atol", "rtol", "dt_geom"))
    assert campaign["config"]["scientific_choices"]["spin_truncation"] == "tabulated"
    names = ("ode_2", "ode_0", "ode_1", "grid_1", "grid_2") if args.part == "levels" else (
        "ode_2", "grid_1", "grid_2")
    source_before = q.source_snapshot()
    output = dict(schema="diffgw.seobnrv4.chain-residual-probe.v1", status="running",
                  scope=dict(q.SCOPE, development_only=True),
                  part=args.part, row=row, plan_sha256=q.sha256(HERE / "PLAN.md"),
                  prior_gate_sha256=q.sha256(GATE),
                  source_manifest_sha256=q.canonical_hash(source_before),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), runs=[])
    original_chain = driver._diagnostic_chain_rule_matching
    original_attach = driver.attach_seobnrv4_22_numerical_compatibility
    direct = []
    value_rows = []

    def capture_chain(time_axis, state, match_time, field, ham, rr, spin_v6):
        result = original_chain(time_axis, state, match_time, field, ham, rr,
                                spin_v6)
        match_state = natural_cubic_spline(time_axis, state).evaluate(match_time)
        omega = field(match_state)[1]
        mode, _ = published_mode22_and_frequency(
            match_state, ham, rr, spin_v6=spin_v6, include_pan_v8=False,
            include_positive_chi_540=False, retain_pan_cubic_spin=True)
        amplitude = mode.abs()
        basis, phase_basis = nqc_22_basis(match_state[0], match_state[2], omega)
        direct.append(dict(state=floats(match_state).tolist(),
                           value=floats(torch.cat((amplitude.reshape(1),
                                                   amplitude * basis, phase_basis))).tolist(),
                           local_match_time_M=float(match_time.detach())))
        return result

    def capture_attach(*positional, **keyword):
        result = original_attach(*positional, **keyword)
        if keyword.get("diagnostic_amplitude_derivatives") is not None:
            assert len(direct) == 1
            nqc = result.nqc
            match = result.match_time
            spline = np.concatenate((
                np.atleast_1d(floats(nqc.amplitude_spline.evaluate(match))),
                np.atleast_1d(floats(nqc.product_spline.evaluate(match))),
                np.atleast_1d(floats(nqc.phase_basis_spline.evaluate(match)))))
            continuous = np.asarray(direct[0]["value"])
            active = floats(nqc.calibration.coefficients.amplitude)
            matrix = floats(nqc.amplitude_matching_matrix)
            rhs = matrix @ active
            alternative_matrix = matrix.copy()
            alternative_matrix[0] = continuous[1:4]
            alternative_rhs = rhs.copy()
            alternative_rhs[0] -= continuous[0] - spline[0]
            alternative = np.linalg.solve(alternative_matrix, alternative_rhs)
            value_rows.append(dict(spline=spline.tolist(), direct=continuous.tolist(),
                                   direct_minus_spline=(continuous-spline).tolist(),
                                   active_amplitude=active.tolist(),
                                   direct_value_amplitude=alternative.tolist(),
                                   coefficient_delta=(alternative-active).tolist(),
                                   matrix_condition=float(np.linalg.cond(matrix))))
        return result

    driver._diagnostic_chain_rule_matching = capture_chain
    driver.attach_seobnrv4_22_numerical_compatibility = capture_attach
    raw = {}
    try:
        for name in names:
            direct.clear()
            value_rows.clear()
            start = time.monotonic()
            kwargs = dict(diagnostic_spin_truncation="tabulated",
                          diagnostic_chain_rule_nqc_derivatives=True,
                          return_nqc_coefficients=True, return_accepted_hashes=True)
            if args.part == "control":
                kwargs.update(diagnostic_restart_time_M=165.0,
                              diagnostic_coarse_initial_step_M=0.5)
            run = q.native_run(row, plans[name], settings, **kwargs)
            assert len(direct) == len(value_rows) == 1
            if args.part == "levels":
                for label in ("amplitude", "phase"):
                    assert np.max(np.abs(np.asarray(run["nqc_coefficients"][label])
                                         - prior[name]["nqc_coefficients"][label])) < 1e-8
                assert run["accepted_array_sha256"] == prior[name]["accepted_array_sha256"]
            raw[name] = run
            item = q.summarize_run(run)
            if args.part == "levels":
                assert item["array_sha256"] == prior[name]["array_sha256"]
            item["continuous"] = direct[0]
            item["value_row"] = value_rows[0]
            item["elapsed_seconds"] = time.monotonic() - start
            output["runs"].append(item)
            q.write_evidence(args.json_out, output)
            print(json.dumps(dict(run=name, seconds=item["elapsed_seconds"],
                                  max_value_row_difference=max(abs(np.asarray(
                                      value_rows[0]["direct_minus_spline"]))),
                                  max_coefficient_delta=max(abs(np.asarray(
                                      value_rows[0]["coefficient_delta"]))))), flush=True)
        if args.part == "levels":
            output["ode"] = q.convergence_check(raw["ode_0"], raw["ode_1"],
                                                 raw["ode_2"], settings)
            output["grid"] = q.convergence_check(raw["ode_2"], raw["grid_1"],
                                                  raw["grid_2"], settings)
        else:
            output["grid"] = q.convergence_check(raw["ode_2"], raw["grid_1"],
                                                  raw["grid_2"], settings)
            output["controlled_nested_equal"] = {}
            for left, right in (("ode_2", "grid_1"), ("grid_1", "grid_2")):
                a, b = raw[left], raw[right]
                step = max(a["dt_geom"], b["dt_geom"])
                down_a = a["mode"][::round(step/a["dt_geom"])]
                down_b = b["mode"][::round(step/b["dt_geom"])]
                output["controlled_nested_equal"][f"{left}_to_{right}"] = bool(
                    np.array_equal(down_a, down_b))
        output["source_unchanged"] = source_before == q.source_snapshot()
        output["status"] = "diagnostic_complete"
        q.write_evidence(args.json_out, output)
        assert output["source_unchanged"]
    finally:
        driver._diagnostic_chain_rule_matching = original_chain
        driver.attach_seobnrv4_22_numerical_compatibility = original_attach


if __name__ == "__main__":
    main()
