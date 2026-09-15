#!/usr/bin/env python3
"""Run the clean HM assembly through an explicit, diagnostic-only adapter.

The adapter in this file is installed temporarily over
``tests.parity_harness.get_fd_waveform``.  It implements only the aligned-spin
``IMRPhenomHM`` call needed by the frozen catalog/holdout checks, using the
independent N=61 London reconstruction and the independent D 220 sampler.
The production dispatcher is never edited or routed through this adapter.

The report keeps the parity harness' complete plus/cross, PSD, and detector
projection metrics.  The two canonical HM cases exercise the unchanged strict
catalog and holdout gates.  Four additional fixed cases from the independent
HM QNM diagnostic provide coverage outside those inventories.  The N=61 grid
description remains a hypothesis inferred from published intervals; this file
does not certify inherited coefficient provenance or fit coefficients to
waveform residuals.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EXACT61_SCRIPT_PATH = ROOT / "scripts" / "diagnose_hm_london_exact61.py"
HM_HELPER_PATH = ROOT / "scripts" / "diagnose_hm_qnm_leaver_clean.py"
PARITY_HARNESS_PATH = ROOT / "tests" / "parity_harness.py"
ASSEMBLY_PATH = ROOT / "src" / "torchwave" / "approximants" / "hm_assembly_clean.py"
PHASE_PATH = ROOT / "src" / "torchwave" / "approximants" / "imrphenom_hm_phase_clean.py"
D_SAMPLING_PATH = ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_d_sampling_clean.py"
D_GENERATED_PATH = ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_generated.py"
EXACT_REPORT_PATH = (
    ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification"
    / "london-qnm-fit-exact61" / "report.json"
)
N257_REPORT_PATH = (
    ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification"
    / "london-qnm-fit" / "report.json"
)
OUTPUT_DIR = (
    ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification"
    / "hm-london-exact61-d-sampling"
)

MODE_LABELS = ((2, 2, 0), (2, 1, 0), (3, 3, 0), (3, 2, 0), (4, 4, 0), (4, 3, 0))
MODE_PAIRS = tuple(mode[:2] for mode in MODE_LABELS)
PARITY_GATE = {"complex_relative_l2": 1.0e-6, "mismatch": 1.0e-8}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _load_modules():
    if str((ROOT / "scripts").resolve()) not in sys.path:
        sys.path.insert(0, str((ROOT / "scripts").resolve()))
    exact61 = importlib.import_module("diagnose_hm_london_exact61")
    helper = importlib.import_module("diagnose_hm_qnm_leaver_clean")
    parity = importlib.import_module("parity_harness")
    return exact61, helper, parity


def _canonical_hm_cases(parity: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate and select the original HM rows without rewriting them."""

    catalog = parity.get_canonical_catalog_cases()
    holdout = parity.get_canonical_holdout_cases()
    catalog_ok, catalog_errors = parity.validate_case_inventory(catalog, "catalog")
    holdout_ok, holdout_errors = parity.validate_case_inventory(holdout, "holdout")
    if not catalog_ok or not holdout_ok:
        raise RuntimeError(
            f"frozen inventory changed: catalog={catalog_errors}, holdout={holdout_errors}"
        )
    catalog_hm = [case for case in catalog if case.get("torch_name") == "IMRPhenomHM"]
    holdout_hm = [case for case in holdout if case.get("torch_name") == "IMRPhenomHM"]
    if len(catalog_hm) != 1 or len(holdout_hm) != 1:
        raise RuntimeError(
            f"expected one canonical IMRPhenomHM row in each inventory; "
            f"got catalog={len(catalog_hm)}, holdout={len(holdout_hm)}"
        )
    return catalog_hm[0], holdout_hm[0], {
        "catalog_validated": True,
        "holdout_validated": True,
        "catalog_case_count": len(catalog),
        "holdout_case_count": len(holdout),
        "selected_catalog_id": catalog_hm[0]["id"],
        "selected_holdout_id": holdout_hm[0]["id"],
    }


def _independent_cases(helper: Any) -> list[dict[str, Any]]:
    """Make parity-shaped copies of the four fixed clean HM diagnostic cases."""

    cases = []
    for index, raw in enumerate(helper.CASES, start=1):
        m1, m2, chi1, chi2, distance, inclination, phic, f_ref = (float(x) for x in raw)
        cases.append(
            {
                "id": f"hm_london_exact61_independent_{index}",
                "torch_name": "IMRPhenomHM",
                "lal_name": "IMRPhenomHM",
                "params": {
                    "mass1": m1,
                    "mass2": m2,
                    "chi1": chi1,
                    "chi2": chi2,
                    "f_ref": f_ref,
                    "distance": distance,
                    "inclination": inclination,
                    "phic": phic,
                    "tc": 0.0,
                },
                "grid": {"flow": 20.0, "fhigh": 512.0, "df": 0.25},
                "expected_gate": dict(PARITY_GATE),
                "verification_scope": "independent_case_raw",
                "oracle_type": "lal",
                "oracle_api": "SimInspiralChooseFDWaveform",
                "source_case": "diagnose_hm_qnm_leaver_clean.CASES",
            }
        )
    return cases


class CleanHMAdapter:
    """Diagnostic implementation of the aligned-spin HM waveform call.

    This object intentionally has the same high-level call shape as the
    parity harness' imported ``get_fd_waveform`` function.  It is kept local to
    the diagnostic process and is never installed in TorchWave's dispatcher.
    """

    def __init__(self, exact61: Any, coefficients: Mapping[Any, Sequence[complex]]):
        import lal
        import torch

        self._exact61 = exact61
        self._coefficients = coefficients
        self._fit_helper = exact61._load_fit_helper()
        self._lal = lal
        self._torch = torch
        from torchwave.approximants.hm_assembly_clean import assemble_mode, project_positive_modes
        from torchwave.approximants.imrphenomd import IMRPhenomD
        from torchwave.approximants.kerr_qnm_d_sampling_clean import evaluate_d_qnm_220

        self._assemble_mode = assemble_mode
        self._project_positive_modes = project_positive_modes
        self._d = IMRPhenomD()
        self._evaluate_d_qnm_220 = evaluate_d_qnm_220
        self._join_kwargs, self._join_metadata = exact61._assembly_option_kwargs(assemble_mode)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "type": "explicit_clean_hm_adapter",
            "production_dispatcher_modified": False,
            "patched_symbol": "tests.parity_harness.get_fd_waveform",
            "restore_after_run": True,
            "fit_count": 61,
            "fit_strategy": "n61_main",
            "join_forwarding": self._join_metadata,
            "d_carrier_qnm_source": "evaluate_d_qnm_220(final_spin) scaled by total_mass/final_mass",
            "hm_qnm_source": "exact N=61 MAIN coefficients from the completed clean report",
        }

    def __call__(
        self,
        approximant: str,
        *,
        sample_frequencies,
        f_lower: float | None = None,
        f_final: float | None = None,
        dtype=None,
        **params,
    ):
        torch = self._torch
        lal = self._lal
        if approximant != "IMRPhenomHM":
            raise ValueError(f"explicit adapter only handles IMRPhenomHM, got {approximant!r}")
        real_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float64
        frequencies = torch.as_tensor(sample_frequencies, dtype=real_dtype)
        eval_frequencies = torch.where(
            frequencies > 0.0, frequencies, torch.ones_like(frequencies)
        )

        m1 = float(params["mass1"])
        m2 = float(params["mass2"])
        chi1 = float(params.get("chi1", params.get("spin1z", 0.0)))
        chi2 = float(params.get("chi2", params.get("spin2z", 0.0)))
        distance = float(params.get("distance", 100.0))
        inclination = float(params.get("inclination", 0.0))
        phic = float(params.get("phic", 0.0))
        tc = float(params.get("tc", 0.0))
        f_ref_raw = params.get("f_ref", None)
        f_ref = None if f_ref_raw is None else float(f_ref_raw)

        total_mass = m1 + m2
        mass_seconds = total_mass * lal.MTSUN_SI
        eta = m1 * m2 / total_mass**2
        delta = (m1 - m2) / total_mass
        chi_s = 0.5 * (chi1 + chi2)
        chi_a = 0.5 * (chi1 - chi2)
        d_args = [torch.tensor(value, dtype=real_dtype) for value in (m1, m2, chi1, chi2)]
        final_mass, final_spin, _, _ = self._d.remnant_properties(*d_args)
        final_mass_value = float(final_mass)
        final_spin_value = float(final_spin)

        # Keep the D remnant mass/spin source opaque.  The absolute carrier
        # controls are sampled independently in final-mass units and then
        # converted to the total-mass convention used by assemble_mode.
        d_fr_final, d_fd_final = self._evaluate_d_qnm_220(final_spin)
        total_to_final = total_mass / final_mass_value
        d_fr22 = float(d_fr_final * total_to_final)
        d_fd22 = float(d_fd_final * total_to_final)

        chi_pn = (m1 * chi1 + m2 * chi2) / total_mass - 38.0 * eta / 113.0 * (chi1 + chi2)
        a4 = float(
            self._d.eval_table_v(
                "alpha4", torch.tensor(eta, dtype=real_dtype), torch.tensor(chi_pn, dtype=real_dtype)
            )
        )
        a5 = float(
            self._d.eval_table_v(
                "alpha5", torch.tensor(eta, dtype=real_dtype), torch.tensor(chi_pn, dtype=real_dtype)
            )
        )
        slope = float(self._d.compute_slope_peak(*d_args))

        def carrier_positive(x):
            return self._d(
                x / mass_seconds,
                m1,
                m2,
                chi1,
                chi2,
                distance=distance,
                inclination=0.0,
                phic=0.0,
                tc=0.0,
                f_ref=f_ref,
                return_phase=True,
            )[2]

        harmonic_22 = complex(lal.SpinWeightedSphericalHarmonic(0.0, 0.0, -2, 2, 2))

        def amplitude_22(x):
            hp, hc = self._d(
                x / mass_seconds,
                m1,
                m2,
                chi1,
                chi2,
                distance=distance,
                inclination=0.0,
                phic=0.0,
                tc=0.0,
                f_ref=None,
            )
            return torch.abs((hp + 1j * hc) / harmonic_22)

        q22 = self._exact61._fit_qnm(
            self._fit_helper, self._coefficients, (2, 2), final_spin_value, total_mass, final_mass_value
        )
        qnm: dict[tuple[int, int], tuple[float, float]] = {
            mode: self._exact61._fit_qnm(
                self._fit_helper,
                self._coefficients,
                mode,
                final_spin_value,
                total_mass,
                final_mass_value,
            )
            for mode in MODE_PAIRS
        }
        harmonic = lambda phi, theta, spin, ell, m: lal.SpinWeightedSphericalHarmonic(
            float(theta), float(phi), int(spin), int(ell), int(m)
        )

        modes = {}
        x = eval_frequencies * mass_seconds
        for mode in MODE_PAIRS:
            mode_qnm = qnm[mode]
            gauge = 0.0 if f_ref is None else ((mode[1] / 2.0) - 1.0) * slope * mass_seconds * f_ref
            modes[mode] = self._assemble_mode(
                x,
                mode,
                eta=eta,
                delta=delta,
                chi_s=chi_s,
                chi_a=chi_a,
                carrier_positive=carrier_positive,
                amplitude_22=amplitude_22,
                amp_qnm_22=q22[0],
                amp_qnm_lm=mode_qnm[0],
                phase_qnm_22=q22[0],
                phase_qnm_lm=mode_qnm[0],
                phase_damp_22=q22[1],
                phase_damp_lm=mode_qnm[1],
                alpha4=a4,
                alpha5=a5,
                chi_pn=chi_pn,
                phic=phic,
                reference_gauge=gauge,
                use_eq14=True,
                base_fr22=d_fr22,
                base_fd22=d_fd22,
                **self._join_kwargs,
            )
        hp, hc = self._project_positive_modes(modes, inclination, harmonic)
        if tc:
            shift = torch.exp(-2j * torch.pi * frequencies * tc)
            hp = hp * shift
            hc = hc * shift

        zero_mask = frequencies <= 0.0
        if f_lower is not None:
            zero_mask = zero_mask | (frequencies < float(f_lower))
        if f_final is not None:
            zero_mask = zero_mask | (frequencies >= float(f_final))
        hp = torch.where(zero_mask, torch.zeros_like(hp), hp)
        hc = torch.where(zero_mask, torch.zeros_like(hc), hc)
        return hp.to(torch.complex128), hc.to(torch.complex128)


def _runtime_gradient_check(exact61: Any, coefficients: Mapping[Any, Sequence[complex]]) -> dict[str, Any]:
    """Exercise sampler and assembly gradients with an actual D carrier."""

    import lal
    import torch
    from torchwave.approximants.hm_assembly_clean import assemble_mode
    from torchwave.approximants.imrphenomd import IMRPhenomD
    from torchwave.approximants.kerr_qnm_d_sampling_clean import evaluate_d_qnm_220

    spin = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    sampler_fr, sampler_fd = evaluate_d_qnm_220(spin)
    sampler_loss = sampler_fr + sampler_fd
    sampler_grad = torch.autograd.grad(sampler_loss, spin, allow_unused=False)[0]

    m1, m2, chi1, chi2 = 40.0, 15.0, 0.3, -0.2
    distance = 500.0
    total_mass = m1 + m2
    mass_seconds = total_mass * lal.MTSUN_SI
    eta = m1 * m2 / total_mass**2
    delta = (m1 - m2) / total_mass
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    d = IMRPhenomD()
    args = [torch.tensor(value, dtype=torch.float64) for value in (m1, m2, chi1, chi2)]
    final_mass, final_spin, _, _ = d.remnant_properties(*args)
    final_mass_value = float(final_mass)
    final_spin_value = float(final_spin)
    q22 = exact61._fit_qnm(exact61._load_fit_helper(), coefficients, (2, 2), final_spin_value, total_mass, final_mass_value)
    q33 = exact61._fit_qnm(exact61._load_fit_helper(), coefficients, (3, 3), final_spin_value, total_mass, final_mass_value)
    d_fr, d_fd = evaluate_d_qnm_220(final_spin)
    d_fr = d_fr * (total_mass / final_mass)
    d_fd = d_fd * (total_mass / final_mass)
    chi_pn = (m1 * chi1 + m2 * chi2) / total_mass - 38.0 * eta / 113.0 * (chi1 + chi2)
    a4 = d.eval_table_v(
        "alpha4", torch.tensor(eta, dtype=torch.float64), torch.tensor(chi_pn, dtype=torch.float64)
    )
    a5 = d.eval_table_v(
        "alpha5", torch.tensor(eta, dtype=torch.float64), torch.tensor(chi_pn, dtype=torch.float64)
    )
    slope = d.compute_slope_peak(*args)
    # Include one point below and two above the 0.5*fRD_D C1 join.  The
    # linear real/imaginary loss below is phase-sensitive, so the explicit
    # carrier QNM inputs are exercised by autograd rather than hidden by a
    # magnitude-only loss.
    frequencies = torch.tensor([100.0, 180.0, 320.0], dtype=torch.float64)
    x = (frequencies * mass_seconds).requires_grad_()
    base_fr = d_fr.detach().clone().requires_grad_()
    base_fd = d_fd.detach().clone().requires_grad_()

    def carrier(y):
        return d(
            y / mass_seconds,
            m1,
            m2,
            chi1,
            chi2,
            distance=distance,
            inclination=0.0,
            phic=0.0,
            tc=0.0,
            f_ref=20.0,
            return_phase=True,
        )[2]

    harmonic_22 = complex(lal.SpinWeightedSphericalHarmonic(0.0, 0.0, -2, 2, 2))

    def amplitude(y):
        hp, hc = d(
            y / mass_seconds,
            m1,
            m2,
            chi1,
            chi2,
            distance=distance,
            inclination=0.0,
            phic=0.0,
            tc=0.0,
            f_ref=None,
        )
        return torch.abs((hp + 1j * hc) / harmonic_22)

    assembled = assemble_mode(
        x,
        (3, 3),
        eta=eta,
        delta=delta,
        chi_s=chi_s,
        chi_a=chi_a,
        carrier_positive=carrier,
        amplitude_22=amplitude,
        amp_qnm_22=q22[0],
        amp_qnm_lm=q33[0],
        phase_qnm_22=q22[0],
        phase_qnm_lm=q33[0],
        phase_damp_22=q22[1],
        phase_damp_lm=q33[1],
        alpha4=a4,
        alpha5=a5,
        chi_pn=chi_pn,
        phic=0.2,
        reference_gauge=0.0,
        use_eq14=True,
        base_fr22=base_fr,
        base_fd22=base_fd,
        amplitude_join=exact61.JOIN_AMPLITUDE,
        phase_join=exact61.JOIN_PHASE,
    )
    assembly_loss = 1.0e24 * (assembled.real + 0.37 * assembled.imag).sum()
    gx, gfr, gfd = torch.autograd.grad(assembly_loss, (x, base_fr, base_fd), allow_unused=False)
    gradients = torch.cat((gx.detach().reshape(-1), gfr.detach().reshape(-1), gfd.detach().reshape(-1)))
    return {
        "sampler": {
            "spin": float(spin.detach()),
            "f_rd_final": float(sampler_fr.detach()),
            "f_damp_final": float(sampler_fd.detach()),
            "loss": float(sampler_loss.detach()),
            "gradient": float(sampler_grad.detach()),
            "finite": bool(torch.isfinite(sampler_grad).item()),
            "nonzero": bool(torch.abs(sampler_grad).item() > 0.0),
        },
        "assembly": {
            "mode": [3, 3],
            "dtype": str(assembled.dtype),
            "loss_scale": 1.0e24,
            "loss": float(assembly_loss.detach()),
            "gradient_x": gx.detach().tolist(),
            "gradient_base_fr22": float(gfr.detach()),
            "gradient_base_fd22": float(gfd.detach()),
            "finite": bool(torch.all(torch.isfinite(gradients)).item()),
            "nonzero": bool(torch.any(torch.abs(gradients) > 0.0).item()),
        },
    }


def _raw_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep concise top-level values while the full result remains in JSON."""

    return {
        "status": result.get("status"),
        "case_id": result.get("case_id"),
        "strict": result.get("strict"),
        "plus_complex_relative_l2": result.get("plus", {}).get("complex_relative_l2"),
        "cross_complex_relative_l2": result.get("cross", {}).get("complex_relative_l2"),
        "plus_flat_mismatch": result.get("plus", {}).get("flat_mismatch"),
        "cross_flat_mismatch": result.get("cross", {}).get("flat_mismatch"),
        "max_projection_complex_relative_l2": max(
            (row.get("complex_relative_l2", float("nan")) for row in result.get("projections", {}).values()),
            default=None,
        ),
        "max_projection_flat_mismatch": max(
            (row.get("flat_mismatch", float("nan")) for row in result.get("projections", {}).values()),
            default=None,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-report", type=Path, default=EXACT_REPORT_PATH)
    parser.add_argument("--n257-report", type=Path, default=N257_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "scripts/diagnose_hm_london_exact61_d_sampling.py": SCRIPT_PATH,
        "scripts/diagnose_hm_london_exact61.py": EXACT61_SCRIPT_PATH,
        "scripts/diagnose_hm_qnm_leaver_clean.py": HM_HELPER_PATH,
        "tests/parity_harness.py": PARITY_HARNESS_PATH,
        "src/torchwave/approximants/hm_assembly_clean.py": ASSEMBLY_PATH,
        "src/torchwave/approximants/imrphenom_hm_phase_clean.py": PHASE_PATH,
        "src/torchwave/approximants/kerr_qnm_d_sampling_clean.py": D_SAMPLING_PATH,
        "src/torchwave/approximants/kerr_qnm_generated.py": D_GENERATED_PATH,
        "exact_N61_fit_report.json": args.exact_report,
        "N257_fit_report.json": args.n257_report,
    }
    source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}
    started = time.monotonic()
    exact61, helper, parity = _load_modules()
    coefficients, fit_metadata = exact61._load_exact_coefficients(args.exact_report)
    catalog_case, holdout_case, inventory = _canonical_hm_cases(parity)
    independent_cases = _independent_cases(helper)
    adapter = CleanHMAdapter(exact61, coefficients)

    report: dict[str, Any] = {
        "status": "running",
        "provenance": {
            "paper": "L. London and E. Fauchon-Jones, arXiv:1708.00404",
            "paper_url": "https://arxiv.org/html/1708.00404",
            "recipe": "fresh London HM assembly through an explicit diagnostic adapter",
            "d_carrier_qnm_source": "evaluate_d_qnm_220(final_spin), scaled by total_mass/final_mass",
            "eq14_carrier_contract": "HM QNM values define the outer frequency map; Eq. (14) uses the D carrier absolute fRD_D and fdamp_D with HM ratios R and T, then C1 completes at 0.5*fRD_D.",
            "adapter_scope": "The adapter replaces only parity_harness.get_fd_waveform for IMRPhenomHM during this process and is restored before exit; production routing is unchanged.",
            "strict_gate_scope": "The unchanged parity gates are run strictly for the original catalog and holdout HM rows. Independent rows are raw diagnostic comparisons.",
            "n61_provenance": "N=61 is a grid metadata hypothesis inferred from published coefficient intervals; this report does not certify inherited coefficient provenance.",
            "clean_room_scope": "No LAL/PyCBC/TEOB implementation or author implementation was copied; LAL is used only as the public binary comparison target through parity_harness.",
        },
        "choices": {
            "fit": fit_metadata,
            "independent_cases": "exactly the four CASES from diagnose_hm_qnm_leaver_clean.py",
            "strict_gate": dict(PARITY_GATE),
            "catalog_case": catalog_case["id"],
            "holdout_case": holdout_case["id"],
            "sampler": "evaluate_d_qnm_220(final_spin), total_mass/final_mass scaling",
            "thresholds_changed": False,
            "coefficient_selection_from_waveform_residuals": False,
        },
        "execution": {
            "host": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "command": [sys.executable, str(SCRIPT_PATH), *(argv if argv is not None else sys.argv[1:])],
            "pid": os.getpid(),
            "log_path": str(args.log_path) if args.log_path else None,
            "python": platform.python_version(),
            "next_check": "inspect report.json after the bounded diagnostic completes",
            "stop_command": f"kill {os.getpid()}",
        },
        "source_hashes_before": source_hashes_before,
        "inventory": inventory,
        "adapter": adapter.metadata,
        "runtime_gradients": None,
        "canonical_strict": [],
        "independent_raw": [],
        "summaries": [],
    }
    _write_json(output_dir / "progress.json", report)

    old_get_fd_waveform = parity.get_fd_waveform
    try:
        parity.get_fd_waveform = adapter
        for label, case in (("catalog", catalog_case), ("holdout", holdout_case)):
            print(f"running strict HM {label} case {case['id']}", flush=True)
            result = parity.run_oracle_comparison(case, strict=True)
            result["adapter"] = adapter.metadata
            report["canonical_strict"].append(result)
            report["summaries"].append({"scope": label, **_raw_summary(result)})
            _write_json(output_dir / "progress.json", report)

        for index, case in enumerate(independent_cases, start=1):
            print(f"running independent HM case {index}/{len(independent_cases)} {case['id']}", flush=True)
            result = parity.run_oracle_comparison(case, strict=False)
            result["adapter"] = adapter.metadata
            report["independent_raw"].append(result)
            report["summaries"].append({"scope": "independent", **_raw_summary(result)})
            _write_json(output_dir / "progress.json", report)
    finally:
        parity.get_fd_waveform = old_get_fd_waveform

    report["runtime_gradients"] = _runtime_gradient_check(exact61, coefficients)
    source_hashes_after = {name: _sha256(path) for name, path in source_paths.items()}
    report["source_hashes_after"] = source_hashes_after
    report["source_hashes_changed_during_run"] = {
        name: (source_hashes_before[name], source_hashes_after[name])
        for name in source_hashes_before
        if source_hashes_before[name] != source_hashes_after[name]
    }
    report["status"] = "complete_diagnostic_only"
    report["elapsed_seconds"] = time.monotonic() - started
    _write_json(output_dir / "report.json", report)
    _write_json(output_dir / "progress.json", report)
    print(output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
