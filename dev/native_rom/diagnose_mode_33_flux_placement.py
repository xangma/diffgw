"""Frozen same-state (3,3) spin-resummation counterfactual for v4 flux."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys

import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_basis_trace import _physics
from torchwave.approximants.eob.factorized_modes import _mass_times_leading_spin_f
from torchwave.approximants.eob.seobnrv4_flux import (
    SEOBNRv4PublishedFlux,
    _higher_kerr_f,
    _nonspin_rho,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/mode_33_flux_placement_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_flux_choices"]
    prior = json.loads(prior_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.mode-33-flux-placement-plan.v1"
            or q.sha256(prior_path) != plan["prior_flux_choices_sha256"]
            or prior["status"] != "diagnostic_complete"
            or [p["target_Momega"] for p in prior["points"]]
            != plan["frequency_targets_Momega"]):
        parser.error("frozen plan or prior flux evidence differs")
    choice = dict(spin_truncation=q.CHOICES["spin_truncation"],
                  retain_pan_cubic_spin=True, mode_21_higher_spin="taylor_f")
    field, _ = _physics(prior["row"])
    ham = field.__self__.hamiltonian
    model = SEOBNRv4PublishedFlux(**choice)
    report = dict(schema="diffgw.seobnrv4.mode-33-flux-placement.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  historical_mode_flux_parity_established=False,
                  plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT/path) for path in (
                      "src/torchwave/approximants/eob/seobnrv4_flux.py",
                      "src/torchwave/approximants/eob/factorized_modes.py")},
                  historical_source=plan["historical_source"],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv),
                  row=prior["row"], choice=choice, points=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for point in prior["points"]:
            s = point["state"]
            state = torch.tensor([s[k] for k in ("r", "phi", "pr_star", "pphi")],
                                 dtype=torch.float64)
            with torch.no_grad():
                quantities = model.from_hamiltonian(state, ham, omega=field(state)[1])
                amplitudes = model._signed_amplitudes(state, **quantities)
                a, b, chi1, chi2 = (quantities[k] for k in
                                    ("x1", "x2", "chi1", "chi2"))
                nu, cs, ca = a*b, (chi1+chi2)/2, (chi1-chi2)/2
                kerr = (1-2*nu)*cs+(a-b)*ca
                v = quantities["omega"].pow(1/3)
                rho_ns = _nonspin_rho(v, nu)[3, 3]
                c = b.square()-a.square()
                leading_f = _mass_times_leading_spin_f(3, 3, v, a, b, chi1, chi2)
                native_residual = c*(rho_ns**3+_higher_kerr_f(v, kerr)[3, 3])+leading_f
                # LALSimIMRSpinEOBFactorizedWaveform.c, v4 non-HM coefficient
                # assignments rho33v4 through rho33v7. Its f33v3 is leading_f.
                historical_rho = (rho_ns+kerr.square()/2*v**4-4*kerr/3*v**5
                                  +5*kerr.square()/36*v**6
                                  +(5297*kerr/2970+kerr**3/3)*v**7)
                historical_residual = c*historical_rho**3+leading_f
                ratio = historical_residual/native_residual
                omega = quantities["omega"]
                mode_flux = (3*omega)**2*amplitudes[3, 3].square()/(8*math.pi)
                native_flux = model.energy_flux(state, **quantities)
                corrected_flux = native_flux+mode_flux*(ratio.square()-1)
                native_torque = -native_flux/(nu*omega)
                corrected_torque = -corrected_flux/(nu*omega)
                baseline = next(row for row in point["choices"]
                                if row["choice"] == choice)
                if not math.isclose(float(native_torque), baseline["torque"], rel_tol=2e-13):
                    raise ValueError("native baseline torque does not replay")
                public_torque = point["sampled_public_torque"]
                report["points"].append(dict(
                    target_Momega=point["target_Momega"],
                    actual_Momega=point["actual_Momega"],
                    native_mode_33_flux_fraction=float(mode_flux/native_flux),
                    historical_to_native_mode_33_amplitude_ratio=float(ratio),
                    native_torque=float(native_torque),
                    mode_33_replaced_torque=float(corrected_torque),
                    sampled_public_torque=public_torque,
                    native_torque_relative_to_public=float(native_torque/public_torque-1),
                    replaced_torque_relative_to_public=float(
                        corrected_torque/public_torque-1)))
            q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
