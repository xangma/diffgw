"""Frozen early aligned-spin same-state field and sampled-torque diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_ode import array, array_hash
from .diagnose_lal_dynamics import COORDINATES, _five_point, _reference
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/early_source_field_plan.json"


def _physics(row: dict, choice: dict):
    parameters = [torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    m1, m2, s1, s2 = parameters
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = HistoricalLogDerivative(x1*x2, s1, s2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(**{
        key: value for key, value in choice.items()
        if key != "hamiltonian_log_derivative"})
    return ham, flux, EOBTrajectoryIntegrator(ham, flux).vector_field


def _point(values: np.ndarray, index: int, step: float, ham, flux, field):
    state = values[1:, index].copy()
    sampled = _five_point(values[1:], index, step)
    with torch.no_grad():
        tensor = torch.from_numpy(state)
        native = array(field(tensor))
        quantities = flux.from_hamiltonian(tensor, ham, omega=torch.as_tensor(native[1]))
        emitted = float(flux.energy_flux(tensor, **quantities))
        nu = float(quantities["x1"]*quantities["x2"])
    torque = -emitted/(nu*native[1])
    if abs(torque-native[3]) > 2e-13*abs(native[3]):
        raise ValueError("native torque and field pphi component disagree")
    error = native-sampled
    return dict(index=index, time_M=float(values[0, index]),
                sampled_frequency_Momega=float(sampled[1]),
                native_frequency_Momega=float(native[1]),
                state=dict(zip(COORDINATES, state.tolist())),
                sampled_derivative=dict(zip(COORDINATES, sampled.tolist())),
                native_derivative=dict(zip(COORDINATES, native.tolist())),
                native_minus_sampled=dict(zip(COORDINATES, error.tolist())),
                relative_difference=dict(zip(
                    COORDINATES, (error/np.maximum(abs(sampled), 1e-15)).tolist())),
                native_emitted_energy_flux=emitted, native_torque=torque,
                native_torque_minus_sampled=torque-float(sampled[3]),
                native_torque_relative_to_sampled=torque/float(sampled[3])-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior"]
    expected_choice = dict(
        hamiltonian_log_derivative="historical", spin_truncation="generic-leading-ell8",
        retain_pan_cubic_spin=True, mode_33_higher_spin="lal_v4_rho",
        mode_21_higher_spin="rho_hybrid", mode_31_higher_spin="lal_v4_rho",
        mode_32_spin_square="lal_v4_zero")
    if (plan["schema"] != "diffgw.seobnrv4.early-source-field-plan.v1"
            or q.sha256(prior_path) != plan["prior_sha256"]
            or plan["row_name"] != "strong_aligned_spin"
            or plan["early_start_Momega"] != 0.03162277660168379
            or plan["public_output_steps_M"] != [0.125, 0.0625]
            or plan["targets_Momega"] != [0.032, 0.035, 0.04, 0.05]
            or plan["native_choice"] != expected_choice
            or plan["release_qualified"] is not False):
        parser.error("frozen plan or prior hash differs")
    prior = json.loads(prior_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["row"] != dict(mass1=30.0, mass2=20.0, chi1=0.8,
                                     chi2=0.8, omega_start=0.04)):
        parser.error("frozen strong aligned-spin row differs")
    pin = prior["lal_binary"]
    if q.lal_binary_identity("7.7.1", "6.2.1") != pin:
        parser.error("pinned public binary identity changed")
    row = dict(prior["row"], omega_start=plan["early_start_Momega"])
    torch.set_num_threads(1)
    ham, flux, field = _physics(row, plan["native_choice"])
    sources = ("dev/native_rom/eob/torch/historical_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "src/torchwave/approximants/eob/integrator.py",
               "dev/native_rom/diagnose_lal_dynamics.py")
    report = dict(
        schema="diffgw.seobnrv4.early-source-field.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        direct_public_flux_parity_established=False,
        public_torque_kind="five_point_derivative_of_public_pphi",
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        script_sha256=q.sha256(__file__),
        source_sha256={name: q.sha256(q.ROOT/name) for name in sources},
        lal_binary=pin, frozen_row=prior["row"], evaluation_row=row,
        native_choice=plan["native_choice"], grids={}, paired_grid_sensitivity=[],
        errors=[], process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                                pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        values = {}
        for step in plan["public_output_steps_M"]:
            print(f"public_output_step_M={step}", flush=True)
            values[step] = _reference(row, step)
            report["grids"][str(step)] = dict(
                shape=list(values[step].shape),
                reference_array_sha256=array_hash(values[step]), points=[])
            q.write_evidence(args.json_out, report)
        coarse, fine = values[0.125], values[0.0625]
        coarse_frequency = _five_point(coarse[2], np.arange(2, coarse.shape[1]-2), 0.125)
        indices = [int(np.argmin(abs(coarse_frequency-target)))+2
                   for target in plan["targets_Momega"]]
        if indices != sorted(set(indices)) or any(
                index < 4 or index >= coarse.shape[1]-4 for index in indices):
            raise ValueError("early target outside distinct safe derivative support")
        for target, coarse_index in zip(plan["targets_Momega"], indices):
            time_m = float(coarse[0, coarse_index])
            fine_index = int(np.argmin(abs(fine[0]-time_m)))
            if (fine_index < 4 or fine_index >= fine.shape[1]-4
                    or abs(float(fine[0, fine_index])-time_m) > 1e-8):
                raise ValueError("public output grids have no shared safe sample")
            points = {
                "0.125": _point(coarse, coarse_index, 0.125, ham, flux, field),
                "0.0625": _point(fine, fine_index, 0.0625, ham, flux, field)}
            for step in ("0.125", "0.0625"):
                report["grids"][step]["points"].append(
                    dict(target_Momega=target, **points[step]))
            c, f = points["0.125"], points["0.0625"]
            sensitivity = dict(
                target_Momega=target, shared_time_M=time_m,
                max_abs_public_state_delta=max(abs(c["state"][key]-f["state"][key])
                                               for key in COORDINATES),
                sampled_coarse_minus_fine={key: c["sampled_derivative"][key]
                                           -f["sampled_derivative"][key]
                                           for key in COORDINATES},
                native_coarse_minus_fine={key: c["native_derivative"][key]
                                          -f["native_derivative"][key]
                                          for key in COORDINATES},
                gap_coarse_minus_fine={key: c["native_minus_sampled"][key]
                                       -f["native_minus_sampled"][key]
                                       for key in COORDINATES},
                sampled_torque_coarse_over_fine_minus_one=(
                    c["sampled_derivative"]["pphi"]
                    /f["sampled_derivative"]["pphi"]-1),
                native_torque_coarse_over_fine_minus_one=(
                    c["native_torque"]/f["native_torque"]-1))
            report["paired_grid_sensitivity"].append(sensitivity)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(target_Momega=target, time_M=time_m,
                                  torque_relative={step: points[step][
                                      "native_torque_relative_to_sampled"]
                                      for step in points},
                                  sampled_torque_coarse_over_fine_minus_one=sensitivity[
                                      "sampled_torque_coarse_over_fine_minus_one"])),
                  flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
