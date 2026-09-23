"""Development-only SEOBNRv4 flux-mode sensitivity diagnostic.

Replay the frozen native row, then evolve with a selected mode removed from
the radiation-reaction flux, or only the independently checked (2,2), (3,3),
and (4,4) modes retained. This is an ablation, not a proposed model or a
qualification test.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics, public_mode


RETAINED = frozenset({(2, 2), (3, 3), (4, 4)})
OMIT = {"omit-21": (2, 1), "omit-32": (3, 2), "omit-55": (5, 5)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whole-waveform", required=True, type=Path)
    parser.add_argument("--peak-anchor", required=True, type=Path)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--selection", choices=("three-mode", *OMIT),
                        default="three-mode")
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
    from torchwave.approximants.eob.factorized_modes import energy_flux_from_modes
    from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
    from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
    from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

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

    def scalar(value):
        return torch.tensor(value, dtype=torch.float64)

    m1, m2 = row["mass1"], row["mass2"]
    nu = m1*m2/(m1+m2)**2
    ham = SEOBNRv4Hamiltonian(
        scalar(nu), scalar(row["chi1"]), scalar(row["chi2"]),
        delta=scalar((m1-m2)/(m1+m2)))
    flux = SEOBNRv4PublishedFlux(
        spin_truncation=q.CHOICES["spin_truncation"],
        retain_pan_cubic_spin=q.CHOICES["include_pan_v8"])
    with torch.no_grad():
        r, pr, j = find_quasicircular_initial_conditions(
            ham, flux, scalar(row["omega_start"]))
        state = torch.stack((r, scalar(0), pr, j))
        quantities = flux.from_hamiltonian(state, ham)
        modes = flux.amplitudes(state, **quantities)
        omega = float(quantities["omega"])
        contributions = {f"{ell}{m}": float((m*omega)**2*amp**2/(8*torch.pi))
                         for (ell, m), amp in modes.items()}
        total = sum(contributions.values())
        if abs(total-float(flux.energy_flux(state, **quantities))) > 1e-15:
            raise ValueError("initial mode decomposition differs from full flux")

    selected = (RETAINED if args.selection == "three-mode"
                else frozenset(modes).difference({OMIT[args.selection]}))

    def retained_flux(self, state, **quantities):
        modes = self._signed_amplitudes(state, **quantities)
        omega_tensor = torch.as_tensor(quantities["omega"], dtype=state.dtype,
                                       device=state.device)
        # Keep the full model's mode order so the intervention is only omission.
        return energy_flux_from_modes(
            {key: amplitude for key, amplitude in modes.items() if key in selected},
            omega_tensor)

    with patch.object(SEOBNRv4PublishedFlux, "energy_flux", retained_flux):
        ablated = q.native_run(row, level, settings, **kwargs)
    ablated_metric = compare(ablated)
    weights = {mode: value/total for mode, value in contributions.items()}
    output = dict(schema="diffgw.seobnrv4.flux-subset-ablation.v1",
                  scope=q.SCOPE, status="diagnostic_complete",
                  whole_waveform_parity_validated=False,
                  row=row, level=level, settings=asdict(settings), numerical_choice=choice,
                  whole_waveform_sha256=q.sha256(args.whole_waveform),
                  peak_anchor_sha256=q.sha256(args.peak_anchor),
                  script_sha256=q.sha256(__file__),
                  lal_binary=dict(lal_version=pin["lal_version"],
                                  lalsimulation_version=pin["lalsimulation_version"],
                                  identity_verified=True),
                  method=("Keep only 22, 33, 44" if args.selection == "three-mode"
                          else f"Omit {args.selection.removeprefix('omit-')}")
                  + " in radiation-reaction flux throughout evolution",
                  selection=args.selection,
                  initial_state=dict(r=float(r), pr_star=float(pr), pphi=float(j),
                                     omega=omega, flux=total),
                  initial_flux_fractions=dict(sorted(weights.items(),
                                                     key=lambda item: -item[1])),
                  retained_initial_fraction=sum(weights[f"{ell}{m}"] for ell, m in selected),
                  baseline=dict(array_sha256=baseline_hash, comparison=baseline_metric),
                  ablated=dict(native=q.summarize_run(ablated), comparison=ablated_metric))
    q.write_evidence(args.json_out, output)
    print(json.dumps(dict(status=output["status"],
                          retained_initial_fraction=output["retained_initial_fraction"],
                          baseline_l2=baseline_metric["metrics"]["complex_relative_l2"],
                          ablated_l2=ablated_metric["metrics"]["complex_relative_l2"],
                          evidence=str(args.json_out.resolve()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
