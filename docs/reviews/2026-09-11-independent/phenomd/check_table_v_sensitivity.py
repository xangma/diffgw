"""Quantify waveform sensitivity of IMRPhenomD to rounding of Table V coefficients.

Loads interval radii (half_last_digit) from table-v-printed.json, applies
deterministic perturbations to TABLE_V_COEFFS, and measures normalized
complex L2 and amplitude differences against the unperturbed baseline waveform.

These constructed perturbations assume independent round-to-nearest intervals.
They measure sensitivity, not the actual unpublished coefficients, a statistical
uncertainty, a worst-case bound, or the cause of any oracle discrepancy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from torchwave.approximants import imrphenomd as imrd
from torchwave.approximants.imrphenomd import IMRPhenomD


def load_table_v_data() -> Dict[str, Any]:
    """Load published Table V values and rounding intervals from JSON."""
    json_path = Path(__file__).parent / "table-v-printed.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for name, values in imrd.TABLE_V_COEFFS.items():
        entries = data["rows"][name]
        if tuple(entry["value"] for entry in entries) != tuple(values):
            raise ValueError(f"Published center differs from runtime row {name}")
        if any(not math.isfinite(entry["value"]) or not math.isfinite(entry["half_last_digit"])
               or entry["half_last_digit"] <= 0 for entry in entries):
            raise ValueError(f"Invalid rounding interval for {name}")
    return data


class TableVPerturbationContext:
    """Context manager to safely apply and revert TABLE_V_COEFFS perturbations."""

    def __init__(self, perturbed_coeffs: Dict[str, Tuple[float, ...]]):
        self.perturbed_coeffs = perturbed_coeffs
        self.original_coeffs: Dict[str, Tuple[float, ...]] = {}

    def __enter__(self):
        self.original_coeffs = {k: tuple(v) for k, v in imrd.TABLE_V_COEFFS.items()}
        for k, v in self.perturbed_coeffs.items():
            if k in imrd.TABLE_V_COEFFS:
                imrd.TABLE_V_COEFFS[k] = v
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        imrd.TABLE_V_COEFFS.clear()
        imrd.TABLE_V_COEFFS.update(self.original_coeffs)


def compute_waveform(
    f: torch.Tensor,
    m1: float,
    m2: float,
    s1: float,
    s2: float,
    distance: float = 400.0,
    inclination: float = 0.5,
    phic: float = 0.3,
    tc: float = 0.0,
    f_ref: float = 20.0,
) -> torch.Tensor:
    """Generate plus polarization for given binary parameters."""
    hp, _ = IMRPhenomD.__call__(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=s1,
        chi2=s2,
        distance=distance,
        inclination=inclination,
        phic=phic,
        tc=tc,
        f_ref=f_ref,
        window_type=None,
    )
    return hp


def evaluate_sensitivity(
    f: torch.Tensor,
    case: Dict[str, Any],
    table_v_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Table V coefficient rounding sensitivity for a single binary configuration."""
    m1, m2 = case["m1"], case["m2"]
    s1, s2 = case["s1"], case["s2"]

    # Baseline waveform with published rounded Table V values
    hp_base = compute_waveform(f, m1, m2, s1, s2)
    norm_base = torch.linalg.norm(hp_base).item()
    norm_base_amp = torch.linalg.norm(torch.abs(hp_base)).item()

    row_results: Dict[str, Dict[str, float]] = {}
    rows_data = table_v_data["rows"]

    # 1. Per-row perturbations (+half_last_digit and -half_last_digit)
    for row_name, row_entries in rows_data.items():
        if row_name not in imrd.TABLE_V_COEFFS:
            continue

        base_vals = [entry["value"] for entry in row_entries]
        radii = [entry["half_last_digit"] for entry in row_entries]

        vals_pos = tuple(v + r for v, r in zip(base_vals, radii))
        vals_neg = tuple(v - r for v, r in zip(base_vals, radii))

        # Test positive perturbation
        with TableVPerturbationContext({row_name: vals_pos}):
            hp_pos = compute_waveform(f, m1, m2, s1, s2)
            rel_l2_pos = (torch.linalg.norm(hp_pos - hp_base).item() / norm_base)
            rel_amp_pos = (torch.linalg.norm(torch.abs(hp_pos) - torch.abs(hp_base)).item() / norm_base_amp)

        # Test negative perturbation
        with TableVPerturbationContext({row_name: vals_neg}):
            hp_neg = compute_waveform(f, m1, m2, s1, s2)
            rel_l2_neg = (torch.linalg.norm(hp_neg - hp_base).item() / norm_base)
            rel_amp_neg = (torch.linalg.norm(torch.abs(hp_neg) - torch.abs(hp_base)).item() / norm_base_amp)

        row_results[row_name] = {
            "rel_l2_pos": rel_l2_pos,
            "rel_l2_neg": rel_l2_neg,
            "rel_amp_pos": rel_amp_pos,
            "rel_amp_neg": rel_amp_neg,
            "max_rel_l2": max(rel_l2_pos, rel_l2_neg),
            "max_rel_amp": max(rel_amp_pos, rel_amp_neg),
        }

    # 2. Multi-row simultaneous perturbations
    all_pos: Dict[str, Tuple[float, ...]] = {}
    all_neg: Dict[str, Tuple[float, ...]] = {}
    alternating: Dict[str, Tuple[float, ...]] = {}

    for row_name, row_entries in rows_data.items():
        if row_name not in imrd.TABLE_V_COEFFS:
            continue
        base_vals = [entry["value"] for entry in row_entries]
        radii = [entry["half_last_digit"] for entry in row_entries]

        all_pos[row_name] = tuple(v + r for v, r in zip(base_vals, radii))
        all_neg[row_name] = tuple(v - r for v, r in zip(base_vals, radii))
        alternating[row_name] = tuple(
            v + (r if i % 2 == 0 else -r) for i, (v, r) in enumerate(zip(base_vals, radii))
        )

    with TableVPerturbationContext(all_pos):
        hp_all_pos = compute_waveform(f, m1, m2, s1, s2)
        rel_l2_all_pos = (torch.linalg.norm(hp_all_pos - hp_base).item() / norm_base)

    with TableVPerturbationContext(all_neg):
        hp_all_neg = compute_waveform(f, m1, m2, s1, s2)
        rel_l2_all_neg = (torch.linalg.norm(hp_all_neg - hp_base).item() / norm_base)

    with TableVPerturbationContext(alternating):
        hp_alt = compute_waveform(f, m1, m2, s1, s2)
        rel_l2_alt = (torch.linalg.norm(hp_alt - hp_base).item() / norm_base)

    simultaneous_max_l2 = max(rel_l2_all_pos, rel_l2_all_neg, rel_l2_alt)

    return {
        "case": case,
        "row_results": row_results,
        "rel_l2_all_pos": rel_l2_all_pos,
        "rel_l2_all_neg": rel_l2_all_neg,
        "rel_l2_alt": rel_l2_alt,
        "simultaneous_max_l2": simultaneous_max_l2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    print("=" * 80)
    print("IMRPhenomD Table V Coefficient Rounding Sensitivity Analysis")
    print("=" * 80)

    table_v_data = load_table_v_data()
    print(f"Loaded Table V metadata from: {table_v_data.get('source', 'arXiv:1508.07253v2')}")
    print(f"Number of rows in JSON: {len(table_v_data['rows'])}")
    print(f"Number of rows in IMRPhenomD.TABLE_V_COEFFS: {len(imrd.TABLE_V_COEFFS)}")

    # Frequency grid: 20 Hz to 1024 Hz with df = 0.25 Hz
    f = torch.arange(20.0, 1024.0 + 0.25, 0.25, dtype=torch.float64)

    test_cases = [
        {"name": "Canonical (30, 20, 0.3, -0.2)", "m1": 30.0, "m2": 20.0, "s1": 0.3, "s2": -0.2},
        {"name": "Equal-mass High-spin (30, 30, 0.8, 0.8)", "m1": 30.0, "m2": 30.0, "s1": 0.8, "s2": 0.8},
        {"name": "High-mass-ratio (70, 10, -0.5, 0.2)", "m1": 70.0, "m2": 10.0, "s1": -0.5, "s2": 0.2},
        {"name": "Equal-mass Zero-spin (20, 20, 0.0, 0.0)", "m1": 20.0, "m2": 20.0, "s1": 0.0, "s2": 0.0},
    ]

    all_case_results = []
    for case in test_cases:
        res = evaluate_sensitivity(f, case, table_v_data)
        all_case_results.append(res)

    # Print detailed row breakdown for the canonical case
    canon_res = all_case_results[0]
    print("\n" + "-" * 80)
    print(f"Per-Row Rounding Sensitivity for {canon_res['case']['name']}")
    print("-" * 80)
    print(f"{'Row Name':<12} | {'Rel L2 (+rad)':<15} | {'Rel L2 (-rad)':<15} | {'Max Rel L2':<15} | {'Max Rel Amp':<15}")
    print("-" * 80)

    for row_name, metrics in canon_res["row_results"].items():
        print(
            f"{row_name:<12} | "
            f"{metrics['rel_l2_pos']:<15.4e} | "
            f"{metrics['rel_l2_neg']:<15.4e} | "
            f"{metrics['max_rel_l2']:<15.4e} | "
            f"{metrics['max_rel_amp']:<15.4e}"
        )

    # Print summary across configurations
    print("\n" + "=" * 80)
    print("Multi-Configuration Sensitivity Summary")
    print("=" * 80)
    print(f"{'Configuration':<40} | {'Max Single-Row L2':<18} | {'Simultaneous L2':<18} | {'Exceeds 1e-6?'}")
    print("-" * 80)

    strict_tolerance = 1.0e-6
    for res in all_case_results:
        case_name = res["case"]["name"]
        max_single = max(m["max_rel_l2"] for m in res["row_results"].values())
        simul = res["simultaneous_max_l2"]
        exceeds = "YES" if simul > strict_tolerance else "NO"
        print(f"{case_name:<40} | {max_single:<18.4e} | {simul:<18.4e} | {exceeds}")

    # Finite, deterministic sensitivity observations
    print("\n" + "=" * 80)
    print("Observed sensitivity (not an error bound)")
    print("=" * 80)
    max_observed_l2 = max(res["simultaneous_max_l2"] for res in all_case_results)
    max_single_observed = max(
        max(m["max_rel_l2"] for m in res["row_results"].values())
        for res in all_case_results
    )

    print(f"1. Maximum complex relative L2 deviation from single-row rounding: {max_single_observed:.4e}")
    print(f"2. Maximum complex relative L2 deviation from simultaneous rounding: {max_observed_l2:.4e}")
    print(f"3. Strict catalog acceptance threshold: {strict_tolerance:.1e}")
    print()
    print("Constructed perturbations assume independent round-to-nearest intervals.")
    print("They do not identify the actual calibration values or establish the cause")
    print("of any oracle discrepancy, a statistical uncertainty, or a worst-case bound.")
    if args.json_out:
        args.json_out.write_text(json.dumps({
            "assumption": "independent round-to-nearest, half printed last digit",
            "frequency_hz": {"start": 20.0, "stop": 1024.0, "step": 0.25},
            "dtype": "float64", "window": None,
            "source": table_v_data.get("source"), "cases": all_case_results,
            "max_single_row_l2": max_single_observed,
            "max_simultaneous_l2": max_observed_l2,
            "acceptance_threshold_unchanged": strict_tolerance,
        }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
