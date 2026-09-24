"""Compare frozen public initial states with two native calibration choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import qualify_generator as q
from .diagnose_hamiltonian_source_conventions import source_coefficients
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative
from .eob.torch.source_calibration_hamiltonian import (
    SOURCE_V4_COEFFICIENTS, SourceCalibratedHistoricalHamiltonian,
)
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_calibration_initial_plan.json"


def _state(row, omega, flux, hamiltonian_type, radial_step):
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)
    m1, m2, s1, s2 = (tensor(row[key]) for key in
                      ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = hamiltonian_type(x1*x2, s1, s2, delta=x1-x2)
    r, pr_star, pphi = find_quasicircular_initial_conditions(
        ham, flux, tensor(omega))
    xi = ham.metric_potentials(r)[2]
    pstar = (xi.detach()*radial_step).requires_grad_(True)
    h_eff = ham.effective_hamiltonian(r, pstar, pphi)
    h_real = ham.real_hamiltonian_from_eff(h_eff)
    radial_slope = xi*torch.autograd.grad(h_eff, pstar)[0]/h_real/radial_step
    zero = tensor(0.).requires_grad_(True)
    h_eff_zero = ham.effective_hamiltonian(r, zero, pphi)
    h_real_zero = ham.real_hamiltonian_from_eff(h_eff_zero)
    first = torch.autograd.grad(h_eff_zero, zero, create_graph=True)[0]/h_real_zero
    analytic_slope = xi.square()*torch.autograd.grad(first, zero)[0]
    finite_pr_star = pr_star*analytic_slope/radial_slope
    return dict(r=float(r.detach()), pr_star=float(pr_star.detach()),
                pphi=float(pphi.detach()),
                finite_radial_response_pr_star=float(finite_pr_star.detach()),
                finite_over_analytic_radial_slope=float(
                    (radial_slope/analytic_slope).detach()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    pre_path = q.ROOT / plan["public_initial_evidence"]
    source_plan_path = q.ROOT / plan["source_convention_plan"]
    if (plan["schema"] != "diffgw.seobnrv4.source-calibration-initial-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_indices"] != [0, 1]
            or plan["radial_step"] != 0.001
            or q.sha256(pre_path) != plan["public_initial_evidence_sha256"]
            or q.sha256(source_plan_path) != plan["source_convention_plan_sha256"]):
        parser.error("frozen plan or input differs")
    pre = json.loads(pre_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    if pre["status"] != "diagnostic_complete" or len(pre["results"]) != 2:
        parser.error("public initial evidence is incomplete")
    parsed = source_coefficients(args.source_root, source_plan)
    embedded = {(name, i, j): value
                for name, rows in SOURCE_V4_COEFFICIENTS.items()
                for i, row in enumerate(rows) for j, value in enumerate(row)}
    if parsed != embedded:
        parser.error("embedded calibration differs from pinned source")
    torch.set_num_threads(1)
    flux = SEOBNRv4PublishedFlux(**plan["flux_choice"])
    report = dict(
        schema="diffgw.seobnrv4.source-calibration-initial.v1",
        scope=q.SCOPE, status="diagnostic_complete", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        public_initial_evidence_sha256=q.sha256(pre_path),
        source_coefficient_header_sha256=source_plan["source_coefficient_header_sha256"],
        source_sha256={name: q.sha256(q.ROOT/name) for name in (
            "src/torchwave/approximants/eob/initial_conditions.py",
            "src/torchwave/approximants/eob/seobnrv4_flux.py",
            "dev/native_rom/eob/torch/source_calibration_hamiltonian.py")},
        flux_choice=plan["flux_choice"], radial_step=plan["radial_step"], rows=[])
    for index in plan["row_indices"]:
        case = pre["results"][index]
        public = case["reference_initial"]
        choices = {
            "published": _state(case["row"], case["native_start_Momega"],
                                flux, HistoricalLogDerivative, plan["radial_step"]),
            "lal_v4_rounded": _state(
                case["row"], case["native_start_Momega"], flux,
                SourceCalibratedHistoricalHamiltonian, plan["radial_step"]),
        }
        for values in choices.values():
            values["minus_public"] = {key: values[key]-public[key]
                                      for key in ("r", "pr_star", "pphi")}
            values["finite_response_minus_public_pr_star"] = (
                values["finite_radial_response_pr_star"]-public["pr_star"])
        report["rows"].append(dict(
            row=case["row"], public_initial=public,
            native_start_Momega=case["native_start_Momega"],
            sampled_public_start_Momega=case["reference_start_Momega"],
            choices=choices))
    q.write_evidence(args.json_out, report)
    print(json.dumps({str(i): {name: values["minus_public"]
                              for name, values in row["choices"].items()}
                      for i, row in enumerate(report["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
