"""Compare frozen native QNM values with the pinned public v4 binary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import lal
import lalsimulation as lalsim
import torch

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q4_q8_qnm_parity_plan.json"


def _public_qnm(row, *, final_mass_solar=None, final_spin=None):
    vector = lal.CreateCOMPLEX16Vector(1)
    if final_mass_solar is None:
        status = lalsim.SimIMREOBGenerateQNMFreqV2(
            vector, row["mass1"], row["mass2"],
            [0., 0., row["chi1"]], [0., 0., row["chi2"]],
            2, 2, 1, lalsim.SEOBNRv4)
    else:
        status = lalsim.SimIMREOBGenerateQNMFreqV2fromFinal(
            vector, final_mass_solar, final_spin, 2, 2, 1)
    if status != 0:
        raise ValueError(f"public QNM call failed: {status}")
    value = complex(vector.data[0]) * (row["mass1"]+row["mass2"]) * lal.MTSUN_SI
    return dict(frequency_M=value.real, damping_M=value.imag)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    attachment_path = q.ROOT / plan["q4_attachment_inputs"]
    qnm_path = q.ROOT / plan["qnm_source"]
    if (plan["schema"] != "diffgw.seobnrv4.q4-q8-qnm-parity-plan.v1"
            or plan["rows"] != ["q4_mixed_spin", "q8_high_aligned"]
            or plan["level"] != "ode_2"
            or plan["absolute_frequency_budget_M"] != 1e-12
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(attachment_path) != plan["q4_attachment_inputs_sha256"]
            or q.sha256(qnm_path) != plan["qnm_source_sha256"]):
        parser.error("frozen plan or inputs differ")
    campaign = json.loads(campaign_path.read_text())
    attachment = json.loads(attachment_path.read_text())
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or attachment["status"] != "diagnostic_complete"
            or attachment["row"] != campaign["rows"]["q4_mixed_spin"]["row"]
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("frozen campaign, attachment, or public binary differs")
    report = dict(schema="diffgw.seobnrv4.q4-q8-qnm-parity.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  campaign_sha256=q.sha256(campaign_path),
                  q4_attachment_inputs_sha256=q.sha256(attachment_path),
                  qnm_source_sha256=q.sha256(qnm_path), lal_binary=pin,
                  rows={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention

        for name in plan["rows"]:
            item = campaign["rows"][name]
            row, native = item["row"], item["runs"][plan["level"]]["native"]
            final_mass, final_spin = native["final_mass_fraction"], native["final_spin"]
            omega, gamma = signed_qnm_220_table_convention(torch.tensor(final_spin, dtype=torch.float64))
            native_qnm = dict(frequency_M=float(omega/final_mass),
                              damping_M=float(gamma/final_mass))
            if name == "q4_mixed_spin":
                captured = attachment["native"]["boundary_inputs"]
                if (native_qnm["frequency_M"] != captured["qnm_frequency"]
                        or native_qnm["damping_M"] != captured["qnm_damping"]):
                    raise ValueError("native q4 QNM differs from captured boundary")
            public_binary = _public_qnm(row)
            public_same_remnant = _public_qnm(
                row, final_mass_solar=final_mass*(row["mass1"]+row["mass2"]),
                final_spin=final_spin)
            differences = {key: abs(public_binary[key]-native_qnm[key])
                           for key in native_qnm}
            same_remnant_differences = {key: abs(public_same_remnant[key]-native_qnm[key])
                                        for key in native_qnm}
            passed = max(*differences.values(), *same_remnant_differences.values()) <= plan["absolute_frequency_budget_M"]
            report["rows"][name] = dict(row=row, final_mass_fraction=final_mass,
                                        final_spin=final_spin, native=native_qnm,
                                        public_binary=public_binary,
                                        public_at_native_remnant=public_same_remnant,
                                        binary_absolute_difference_M=differences,
                                        same_remnant_absolute_difference_M=same_remnant_differences,
                                        qnm_parity_passed=passed)
            if not passed:
                raise ValueError(f"QNM parity failed for {name}")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
