"""Development-only SEOBNRv4 initial radial-momentum ablation.

Replay a frozen native row and pinned public binary, then set only the solved
initial p_r* to zero. This is a sensitivity test, not a model alternative.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics, public_mode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole-waveform", required=True, type=Path)
    parser.add_argument("--peak-anchor", required=True, type=Path)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)

    prior = json.loads(args.whole_waveform.read_text())
    peaks = json.loads(args.peak_anchor.read_text())
    index = args.row_index
    if (prior.get("status") != "diagnostic_complete"
            or peaks.get("status") != "diagnostic_complete"
            or peaks.get("prior_whole_waveform", {}).get("sha256") != q.sha256(args.whole_waveform)
            or not 0 <= index < len(prior["rows"])
            or prior["rows"][index] != peaks["results"][index]["row"]):
        parser.error("incompatible frozen diagnostic evidence")
    pin = peaks["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary differs from frozen evidence")

    import torch
    import torchwave.approximants.eob.initial_conditions as ic

    torch.set_num_threads(1)
    row = prior["rows"][index]
    level = prior["level"]
    settings = q.Settings(**prior["settings"])
    choice = prior["numerical_choice"]
    kwargs = dict(diagnostic_hermite_resampling=choice["hermite_resampling"],
                  diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                  diagnostic_restart_time_M=choice["fixed_restart_time_M"])

    def compare(run):
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        return fitted_metrics(nt, nm, rt, rm, mass_seconds, "quadratic_power")

    baseline = q.native_run(row, level, settings, **kwargs)
    baseline_hash = q.summarize_run(baseline)["array_sha256"]
    if baseline_hash != prior["results"][index]["native"]["array_sha256"]:
        raise ValueError("native replay differs from frozen waveform")
    baseline_metric = compare(baseline)
    expected = peaks["results"][index]["fitted_peaks"]["quadratic_power"]["metrics"]
    if abs(baseline_metric["metrics"]["complex_relative_l2"]
           - expected["complex_relative_l2"]) > 1e-12:
        raise ValueError("baseline comparison differs from frozen peak diagnostic")

    original = ic.find_quasicircular_initial_conditions
    initial = {}

    def zero_radial_momentum(*positional, **named):
        r, pr_star, pphi = original(*positional, **named)
        initial.update(r=float(r), pr_star=float(pr_star), pphi=float(pphi))
        return r, pr_star * 0, pphi

    with patch.object(ic, "find_quasicircular_initial_conditions", zero_radial_momentum):
        ablated = q.native_run(row, level, settings, **kwargs)
    if not initial or initial["pr_star"] == 0:
        raise ValueError("initial radial-momentum ablation was not applied")
    ablated_metric = compare(ablated)
    output = dict(schema="diffgw.seobnrv4.initial-momentum-ablation.v1",
                  scope=q.SCOPE, status="diagnostic_complete",
                  whole_waveform_parity_validated=False,
                  row=row, level=level, settings=asdict(settings), numerical_choice=choice,
                  whole_waveform_sha256=q.sha256(args.whole_waveform),
                  peak_anchor_sha256=q.sha256(args.peak_anchor),
                  script_sha256=q.sha256(__file__),
                  lal_binary=dict(lal_version=pin["lal_version"],
                                  lalsimulation_version=pin["lalsimulation_version"],
                                  identity_verified=True),
                  method="Zero only the solved initial p_r*; retain r and p_phi",
                  original_initial_state=initial,
                  baseline=dict(array_sha256=baseline_hash, comparison=baseline_metric),
                  ablated=dict(native=q.summarize_run(ablated), comparison=ablated_metric))
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"],
                          baseline_l2=baseline_metric["metrics"]["complex_relative_l2"],
                          ablated_l2=ablated_metric["metrics"]["complex_relative_l2"],
                          evidence=str(args.json_out.resolve()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
