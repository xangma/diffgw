"""Recompute the frozen early public states with source-rounded calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_hamiltonian_source_conventions import source_coefficients
from .diagnose_lal_dynamics import COORDINATES
from .eob.torch.source_calibration_hamiltonian import (
    SOURCE_V4_COEFFICIENTS, SourceCalibratedHistoricalHamiltonian,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_calibration_early_field_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / plan[name] for name in
             ("prior_field", "waveform", "source_convention_plan")}
    if (plan["schema"] != "diffgw.seobnrv4.source-calibration-early-field-plan.v1"
            or plan["release_qualified"] is not False
            or any(q.sha256(path) != plan[f"{name}_sha256"]
                   for name, path in paths.items())):
        parser.error("frozen plan or inputs changed")
    prior = json.loads(paths["prior_field"].read_text())
    waveform = json.loads(paths["waveform"].read_text())
    source_plan = json.loads(paths["source_convention_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or waveform["status"] != "diagnostic_complete"
            or not waveform["check"]["numerical_check_passed"]
            or prior["lal_binary"] != waveform["lal_binary"]
            or prior["frozen_row"] != waveform["row"]):
        parser.error("prerequisite evidence is inconsistent")
    parsed = source_coefficients(args.source_root, source_plan)
    embedded = {(name, i, j): value
                for name, rows in SOURCE_V4_COEFFICIENTS.items()
                for i, row in enumerate(rows) for j, value in enumerate(row)}
    if parsed != embedded:
        parser.error("embedded calibration differs from pinned source")
    row = prior["evaluation_row"]
    m1, m2, s1, s2 = (torch.tensor(row[key], dtype=torch.float64)
                      for key in ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = SourceCalibratedHistoricalHamiltonian(x1*x2, s1, s2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(**{
        key: value for key, value in prior["native_choice"].items()
        if key != "hamiltonian_log_derivative"})
    field = EOBTrajectoryIntegrator(ham, flux).vector_field
    result = dict(
        schema="diffgw.seobnrv4.source-calibration-early-field.v1",
        status="diagnostic_complete", release_qualified=False, scope=q.SCOPE,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        input_sha256={name: q.sha256(path) for name, path in paths.items()},
        source_coefficient_header_sha256=source_plan["source_coefficient_header_sha256"],
        source_sha256={name: q.sha256(q.ROOT/name) for name in (
            "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
            "dev/native_rom/eob/torch/historical_hamiltonian.py",
            "src/torchwave/approximants/eob/seobnrv4_flux.py",
            "src/torchwave/approximants/eob/integrator.py")},
        row=row, grids={}, maxima={})
    torch.set_num_threads(1)
    for step, grid in prior["grids"].items():
        points = []
        for point in grid["points"]:
            state = torch.tensor([point["state"][name] for name in COORDINATES],
                                 dtype=torch.float64)
            with torch.no_grad():
                native = field(state).detach().numpy()
            sampled = np.array([point["sampled_derivative"][name]
                                for name in COORDINATES])
            old = np.array([point["native_derivative"][name]
                            for name in COORDINATES])
            relative = (native-sampled)/np.maximum(abs(sampled), 1e-15)
            points.append(dict(
                target_Momega=point["target_Momega"],
                time_M=point["time_M"],
                rounded_native_derivative=dict(zip(COORDINATES, native.tolist())),
                rounded_relative_to_sampled=dict(zip(COORDINATES, relative.tolist())),
                rounded_minus_published_native=dict(zip(
                    COORDINATES, (native-old).tolist())),
                published_relative_to_sampled=point["relative_difference"]))
        result["grids"][step] = dict(public_array_sha256=grid["reference_array_sha256"],
                                    points=points)
        result["maxima"][step] = {
            coordinate: dict(
                published=max(abs(point["published_relative_to_sampled"][coordinate])
                              for point in points),
                rounded=max(abs(point["rounded_relative_to_sampled"][coordinate])
                            for point in points))
            for coordinate in COORDINATES}
    q.write_evidence(args.json_out, result)
    print(json.dumps(result["maxima"], sort_keys=True))


if __name__ == "__main__":
    main()
