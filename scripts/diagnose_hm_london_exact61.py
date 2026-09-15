#!/usr/bin/env python3
"""Bounded HM waveform diagnostic for the independent exact-N=61 QNM fit.

The exact-N=61 coefficients are read, at full JSON precision, from the
separate clean-room reconstruction report.  They are passed through the
fresh equation-level HM assembly and compared with the public HM modes and
polarizations on the four fixed cases used by
``diagnose_hm_qnm_leaver_clean.py``.  The public four-decimal London values
and the original N=257 MAIN coefficients are controls only; no coefficient
set is selected from a waveform residual.

This file is a diagnostic.  It does not alter production coefficients, solve
new roots, call an author fitting implementation, or copy a waveform
implementation.  The public LAL HM callable is used only as the comparison
target through the already-reviewed diagnostic helper.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import inspect
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
HM_HELPER_PATH = ROOT / "scripts" / "diagnose_hm_qnm_leaver_clean.py"
FIT_HELPER_PATH = ROOT / "scripts" / "diagnose_london_qnm_fit_clean.py"
EXACT_FIT_SCRIPT_PATH = ROOT / "scripts" / "diagnose_london_qnm_fit_exact61.py"
ASSEMBLY_PATH = ROOT / "src" / "torchwave" / "approximants" / "hm_assembly_clean.py"
PHASE_PATH = ROOT / "src" / "torchwave" / "approximants" / "imrphenom_hm_phase_clean.py"
PRINTED_PATH = ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_london_published.py"
D_SAMPLING_PATH = ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_d_sampling_clean.py"
D_GENERATED_PATH = ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_generated.py"
EXACT_REPORT_PATH = (
    ROOT
    / "docs"
    / "reviews"
    / "2026-09-13-parity-review"
    / "verification"
    / "london-qnm-fit-exact61"
    / "report.json"
)
N257_REPORT_PATH = (
    ROOT
    / "docs"
    / "reviews"
    / "2026-09-13-parity-review"
    / "verification"
    / "london-qnm-fit"
    / "report.json"
)

MODE_LABELS = ((2, 2, 0), (2, 1, 0), (3, 3, 0), (3, 2, 0), (4, 4, 0), (4, 3, 0))
MODE_PAIRS = tuple(mode[:2] for mode in MODE_LABELS)
JOIN_AMPLITUDE = 0.014
JOIN_PHASE = 0.018
VARIANT_NAMES = ("n61_main", "printed_london", "n257_main")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert NumPy values and complex values to strict JSON values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
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


def _load_hm_helper():
    """Import the allowed public-target diagnostic as a read-only helper."""
    if str(HM_HELPER_PATH.parent) not in sys.path:
        sys.path.insert(0, str(HM_HELPER_PATH.parent))
    return importlib.import_module("diagnose_hm_qnm_leaver_clean")


def _load_fit_helper():
    """Import only the original clean-room fit/evaluation helper."""
    if str(FIT_HELPER_PATH.parent) not in sys.path:
        sys.path.insert(0, str(FIT_HELPER_PATH.parent))
    return importlib.import_module("diagnose_london_qnm_fit_clean")


def _complex_coefficients(raw: Any, *, mode: tuple[int, int, int]) -> list[complex]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"missing coefficients for mode {mode}")
    result: list[complex] = []
    for index, pair in enumerate(raw):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"coefficient {index} for mode {mode} is not [real, imag]")
        value = complex(float(pair[0]), float(pair[1]))
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError(f"non-finite coefficient {index} for mode {mode}")
        result.append(value)
    return result


def _load_exact_coefficients(path: Path) -> tuple[dict[tuple[int, int, int], list[complex]], dict[str, Any]]:
    report = json.loads(path.read_text())
    if report.get("status") not in {"complete", "completed"}:
        raise ValueError(f"exact fit report is not complete: {path}")
    fits: dict[tuple[int, int, int], list[complex]] = {}
    for mode in MODE_LABELS:
        label = ",".join(str(value) for value in mode)
        entry = report.get("modes", {}).get(label, {})
        fit = entry.get("fit_result", {})
        if fit.get("count") != 61 or fit.get("fit", {}).get("strategy") != "main":
            raise ValueError(f"exact report mode {label} is not the declared N=61 MAIN fit")
        fits[mode] = _complex_coefficients(fit.get("coefficients_rectangular"), mode=mode)
    metadata = {
        "path": str(path),
        "sha256": _sha256(path),
        "count": 61,
        "strategy": "main",
        "coefficients_source": "fit_result.coefficients_rectangular, full JSON precision",
        "report_provenance": report.get("provenance", {}),
        "source_hashes_before": report.get("source_hashes_before", {}),
        "source_hashes_after": report.get("source_hashes_after", {}),
        "root_artifacts": {
            label: report["modes"][label].get("raw_artifacts", {})
            for label in (",".join(str(value) for value in mode) for mode in MODE_LABELS)
        },
    }
    if metadata["source_hashes_before"] != metadata["source_hashes_after"]:
        raise ValueError("exact fit report source hashes changed during its reconstruction")
    return fits, metadata


def _load_n257_coefficients(path: Path) -> tuple[dict[tuple[int, int, int], list[complex]], dict[str, Any]]:
    report = json.loads(path.read_text())
    if report.get("status") not in {"complete", "completed"}:
        raise ValueError(f"N=257 fit report is not complete: {path}")
    fits: dict[tuple[int, int, int], list[complex]] = {}
    grid_metadata: dict[str, Any] = {}
    for mode in MODE_LABELS:
        label = ",".join(str(value) for value in mode)
        entries = [
            item
            for item in report.get("modes", {}).get(label, {}).get("grid_results", [])
            if item.get("count") == 257 and item.get("fit", {}).get("strategy") == "main"
        ]
        if len(entries) != 1:
            raise ValueError(f"expected one N=257 MAIN fit for mode {label}, got {len(entries)}")
        entry = entries[0]
        fits[mode] = _complex_coefficients(entry.get("coefficients_rectangular"), mode=mode)
        grid_metadata[label] = {
            "count": entry.get("count"),
            "strategy": entry.get("fit", {}).get("strategy"),
            "training_residual": entry.get("fit", {}).get("training_residual"),
            "printed_comparison": entry.get("printed_comparison"),
        }
    return fits, {
        "path": str(path),
        "sha256": _sha256(path),
        "count": 257,
        "strategy": "main",
        "coefficients_source": "grid_results[count=257].coefficients_rectangular, full JSON precision",
        "report_provenance": report.get("provenance", {}),
        "source_hashes_before": report.get("source_hashes_before", {}),
        "source_hashes_after": report.get("source_hashes_after", {}),
        "per_mode_fit_summary": grid_metadata,
    }


def _assembly_option_kwargs(assemble_mode: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Forward explicit join fractions when the owning assembly exposes them.

    The D carrier QNM values used by Eq. (14) are case-specific and are added
    by :func:`_case` from that case's remnant output.  This helper handles only
    the fixed join fractions, while signature detection keeps the diagnostic
    compatible with assemblies that expose or internalize those joins.
    """
    signature = inspect.signature(assemble_mode)
    aliases = {
        "amp_join": JOIN_AMPLITUDE,
        "amplitude_join": JOIN_AMPLITUDE,
        "amplitude_join_fraction": JOIN_AMPLITUDE,
        "amp_join_fraction": JOIN_AMPLITUDE,
        "phase_join": JOIN_PHASE,
        "phase_join_fraction": JOIN_PHASE,
    }
    kwargs: dict[str, Any] = {}
    forwarded: dict[str, str | None] = {"amplitude": None, "phase": None}
    for name, value in aliases.items():
        if name not in signature.parameters:
            continue
        kwargs[name] = value
        if value == JOIN_AMPLITUDE:
            forwarded["amplitude"] = name
        else:
            forwarded["phase"] = name
    # Eq. (14)'s absolute carrier controls are supplied per case below.  Do
    # not put a neutral value in this shared options dictionary: doing so
    # would silently replace the D carrier with the HM 22 QNM.
    return kwargs, {
        "assemble_mode_signature": str(signature),
        "amplitude_join": JOIN_AMPLITUDE,
        "phase_join": JOIN_PHASE,
        "explicit_forwarded_parameter": forwarded,
        "base_fr22": "per_case_D_carrier" if "base_fr22" in signature.parameters else "unavailable",
        "base_fd22": "per_case_D_carrier" if "base_fd22" in signature.parameters else "unavailable",
    }


def _metrics(got: Any, reference: Any) -> dict[str, float]:
    """Report raw complex, phase, and amplitude errors without a gate."""
    lhs = np.asarray(got, dtype=np.complex128)
    rhs = np.asarray(reference, dtype=np.complex128)
    if lhs.shape != rhs.shape:
        raise ValueError(f"shape mismatch: {lhs.shape} versus {rhs.shape}")
    error = lhs - rhs
    tiny = np.finfo(float).tiny
    relative = np.abs(error) / np.maximum(np.abs(rhs), tiny)
    phase = np.angle(lhs * np.conj(rhs))
    amplitude = np.abs(lhs) - np.abs(rhs)
    norm = max(float(np.linalg.norm(rhs)), tiny)
    return {
        "raw_error_l2": float(np.linalg.norm(error) / norm),
        "raw_error_max_abs": float(np.max(np.abs(error))),
        "raw_error_max_relative": float(np.max(relative)),
        "phase_rms_rad": float(np.sqrt(np.mean(phase * phase))),
        "phase_max_abs_rad": float(np.max(np.abs(phase))),
        "amplitude_error_l2": float(np.linalg.norm(amplitude) / norm),
        "amplitude_error_max_abs": float(np.max(np.abs(amplitude))),
        "amplitude_error_max_relative": float(
            np.max(np.abs(amplitude) / np.maximum(np.abs(rhs), tiny))
        ),
    }


def _fit_qnm(
    fit_helper: Any,
    coefficients: Mapping[tuple[int, int, int], Sequence[complex]],
    mode: tuple[int, int],
    af: float,
    mt: float,
    mf: float,
) -> tuple[float, float]:
    mode3 = (*mode, 0)
    z = fit_helper.evaluate_coefficients(
        np.asarray([af], dtype=float), mode3, coefficients[mode3], "main"
    )[0]
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        raise ValueError(f"non-finite fitted QNM for {mode} at af={af}")
    scale = float(mt) / float(mf)
    return float(z.real / (2.0 * math.pi) * scale), float(abs(z.imag) / (2.0 * math.pi) * scale)


def _case(c: Sequence[float], fit_sets: Mapping[str, Mapping[tuple[int, int, int], Sequence[complex]]], join_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all declared QNM controls for one existing fixed case."""
    import lal
    import torch

    helper = _load_hm_helper()
    fit_helper = _load_fit_helper()
    from torchwave.approximants.hm_assembly_clean import assemble_mode, project_positive_modes
    from torchwave.approximants.imrphenomd import IMRPhenomD
    from torchwave.approximants.kerr_qnm_d_sampling_clean import evaluate_d_qnm_220

    m1, m2, s1, s2, distance, inclination, phic, fref = (float(x) for x in c)
    mt = m1 + m2
    M = mt * helper.MTSUN
    eta = m1 * m2 / mt**2
    delta = (m1 - m2) / mt
    chi_s = (s1 + s2) / 2.0
    chi_a = (s1 - s2) / 2.0
    d = IMRPhenomD()
    args = [torch.tensor(x, dtype=torch.float64) for x in (m1, m2, s1, s2)]
    mf, af, _, _ = d.remnant_properties(*args)
    mf = float(mf)
    af = float(af)
    d_qnm_fr22, d_qnm_fd22 = evaluate_d_qnm_220(torch.tensor(af, dtype=torch.float64))
    carrier_scale = mt / mf
    dfr22 = float(d_qnm_fr22 * carrier_scale)
    dfd22 = float(d_qnm_fd22 * carrier_scale)
    chi_pn = (m1 * s1 + m2 * s2) / mt - 38.0 * eta / 113.0 * (s1 + s2)
    a4 = float(
        d.eval_table_v(
            "alpha4", torch.tensor(eta, dtype=torch.float64), torch.tensor(chi_pn, dtype=torch.float64)
        )
    )
    a5 = float(
        d.eval_table_v(
            "alpha5", torch.tensor(eta, dtype=torch.float64), torch.tensor(chi_pn, dtype=torch.float64)
        )
    )
    slope = float(d.compute_slope_peak(*args))
    scale = mt * lal.MTSUN_SI * (mt * lal.MRSUN_SI) / (distance * 1e6 * lal.PC_SI)

    frequencies = np.asarray(helper.FREQ, dtype=float)
    raw = helper.public_modes(frequencies, c)
    target_modes = {mode: torch.tensor(raw[mode] * scale, dtype=torch.complex128) for mode in MODE_PAIRS}

    def carrier(x: torch.Tensor) -> torch.Tensor:
        return d(
            x / M,
            m1,
            m2,
            s1,
            s2,
            distance=distance,
            inclination=0.0,
            phic=0.0,
            tc=0.0,
            f_ref=fref,
            return_phase=True,
        )[2]

    def amplitude_22(x: torch.Tensor) -> torch.Tensor:
        hp, hc = d(
            x / M,
            m1,
            m2,
            s1,
            s2,
            distance=distance,
            inclination=0.0,
            phic=0.0,
            tc=0.0,
            f_ref=None,
        )
        harmonic_22 = complex(lal.SpinWeightedSphericalHarmonic(0.0, 0.0, -2, 2, 2))
        return torch.abs((hp + 1j * hc) / harmonic_22)

    # project_positive_modes supplies (phi, theta); LAL takes (theta, phi).
    harmonic = lambda phi, theta, spin, ell, m: lal.SpinWeightedSphericalHarmonic(
        float(theta), float(phi), int(spin), int(ell), int(m)
    )
    target_hp, target_hc = project_positive_modes(target_modes, inclination, harmonic)

    qnm_sets: dict[str, dict[tuple[int, int], tuple[float, float]]] = {}
    for name, coefficients in fit_sets.items():
        qnm_sets[name] = {
            mode: _fit_qnm(fit_helper, coefficients, mode, af, mt, mf) for mode in MODE_PAIRS
        }
    qnm_sets["printed_london"] = {
        mode: tuple(float(x) for x in helper.qnm_fit(mode, af, mt, mf)) for mode in MODE_PAIRS
    }

    assembly_parameters = inspect.signature(assemble_mode).parameters
    assembly_kwargs = dict(join_kwargs)
    if "base_fr22" in assembly_parameters:
        assembly_kwargs["base_fr22"] = dfr22
    if "base_fd22" in assembly_parameters:
        assembly_kwargs["base_fd22"] = dfd22

    predictions: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    polarizations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    variant_rows: dict[str, Any] = {}
    for name in VARIANT_NAMES:
        q = qnm_sets[name]
        predicted: dict[tuple[int, int], torch.Tensor] = {}
        for mode in MODE_PAIRS:
            gauge = ((mode[1] / 2.0) - 1.0) * slope * M * fref
            predicted[mode] = assemble_mode(
                torch.tensor(frequencies * M, dtype=torch.float64),
                mode,
                eta=eta,
                delta=delta,
                chi_s=chi_s,
                chi_a=chi_a,
                carrier_positive=carrier,
                amplitude_22=amplitude_22,
                amp_qnm_22=q[(2, 2)][0],
                amp_qnm_lm=q[mode][0],
                phase_qnm_22=q[(2, 2)][0],
                phase_qnm_lm=q[mode][0],
                phase_damp_22=q[(2, 2)][1],
                phase_damp_lm=q[mode][1],
                alpha4=a4,
                alpha5=a5,
                chi_pn=chi_pn,
                phic=phic,
                reference_gauge=gauge,
                use_eq14=True,
                **assembly_kwargs,
            ).detach()
        predicted_np = {mode: value.cpu().numpy().astype(np.complex128) for mode, value in predicted.items()}
        predictions[name] = predicted_np
        hp, hc = project_positive_modes(predicted, inclination, harmonic)
        polarizations[name] = (hp.cpu().numpy().astype(np.complex128), hc.cpu().numpy().astype(np.complex128))

    for name in VARIANT_NAMES:
        mode_rows: dict[str, Any] = {}
        for mode in MODE_PAIRS:
            row: dict[str, Any] = {"vs_public_hm": _metrics(predictions[name][mode], raw[mode] * scale)}
            if name != "printed_london":
                row["vs_printed_london"] = _metrics(
                    predictions[name][mode], predictions["printed_london"][mode]
                )
            mode_rows[str(mode)] = row
        hp_row: dict[str, Any] = {
            "vs_public_hm": _metrics(polarizations[name][0], target_hp.cpu().numpy())
        }
        hc_row: dict[str, Any] = {
            "vs_public_hm": _metrics(polarizations[name][1], target_hc.cpu().numpy())
        }
        if name != "printed_london":
            hp_row["vs_printed_london"] = _metrics(polarizations[name][0], polarizations["printed_london"][0])
            hc_row["vs_printed_london"] = _metrics(polarizations[name][1], polarizations["printed_london"][1])
        variant_rows[name] = {
            "qnm_total_M": q,
            "mode_errors": mode_rows,
            "polarization_errors": {"hp": hp_row, "hc": hc_row},
        }

    return {
        "inputs": list(c),
        "remnant": {
            "final_mass_solar_mass": mf,
            "final_mass_fraction": mf / mt,
            "final_spin": af,
            "D_carrier_f_rd": dfr22,
            "D_carrier_f_damp": dfd22,
            "D_carrier_qnm_source": "evaluate_d_qnm_220(final_spin) scaled by total_mass/final_mass",
            "D_carrier_total_to_final_scale": carrier_scale,
            "slope_peak": slope,
        },
        "assembly_inputs": {
            "eta": eta,
            "delta": delta,
            "chi_s": chi_s,
            "chi_a": chi_a,
            "chi_pn": chi_pn,
            "alpha4": a4,
            "alpha5": a5,
            "mass_seconds": M,
            "public_mass_distance_scale": scale,
        },
        "variants": variant_rows,
        "sample_count": int(len(frequencies)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-report", type=Path, default=EXACT_REPORT_PATH)
    parser.add_argument("--n257-report", type=Path, default=N257_REPORT_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs"
        / "reviews"
        / "2026-09-13-parity-review"
        / "verification"
        / "hm-london-exact61",
    )
    parser.add_argument("--log-path", type=Path, help="path recorded for the redirected job log")
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_coefficients, exact_metadata = _load_exact_coefficients(args.exact_report)
    n257_coefficients, n257_metadata = _load_n257_coefficients(args.n257_report)
    fit_sets = {
        "n61_main": exact_coefficients,
        "n257_main": n257_coefficients,
    }
    helper = _load_hm_helper()
    from torchwave.approximants.hm_assembly_clean import assemble_mode

    join_kwargs, join_metadata = _assembly_option_kwargs(assemble_mode)
    source_paths = {
        "scripts/diagnose_hm_london_exact61.py": SCRIPT_PATH,
        "scripts/diagnose_hm_qnm_leaver_clean.py": HM_HELPER_PATH,
        "scripts/diagnose_london_qnm_fit_clean.py": FIT_HELPER_PATH,
        "scripts/diagnose_london_qnm_fit_exact61.py": EXACT_FIT_SCRIPT_PATH,
        "src/torchwave/approximants/hm_assembly_clean.py": ASSEMBLY_PATH,
        "src/torchwave/approximants/imrphenom_hm_phase_clean.py": PHASE_PATH,
        "src/torchwave/approximants/kerr_qnm_london_published.py": PRINTED_PATH,
        "src/torchwave/approximants/kerr_qnm_d_sampling_clean.py": D_SAMPLING_PATH,
        "src/torchwave/approximants/kerr_qnm_generated.py": D_GENERATED_PATH,
        "exact_N61_fit_report.json": args.exact_report,
        "N257_fit_report.json": args.n257_report,
    }
    source_hashes_before = {name: _sha256(path) for name, path in source_paths.items()}
    started = time.monotonic()
    command = [sys.executable, str(SCRIPT_PATH), *(argv if argv is not None else sys.argv[1:])]

    report: dict[str, Any] = {
        "provenance": {
            "paper": "L. London and E. Fauchon-Jones, arXiv:1708.00404",
            "paper_url": "https://arxiv.org/html/1708.00404",
            "recipe": "fresh HM assembly using the published London equations and explicit positive-m projection",
            "eq14_carrier_contract": "HM QNM values define the outer frequency map; Eq. (14) uses the D carrier's absolute fRD_D and fdamp_D, with HM ratios R=f22_HM/flm_HM and T=fdlm_HM/fd22_HM, and C1 completion at 0.5*fRD_D.",
            "d_carrier_qnm_source": "evaluate_d_qnm_220(final_spin), scaled by total_mass/final_mass; remnant mass and spin remain the opaque IMRPhenomD outputs.",
            "note": "Regenerated after applying the explicit per-case D-carrier QNM contract; the reported residuals are raw diagnostic measurements and do not select coefficients or claim a strict parity gate.",
            "public_target": "existing diagnose_hm_qnm_leaver_clean.py public HM callable and polarizations",
            "clean_room_scope": "No LAL/PyCBC/TEOB implementation or author implementation was copied; LAL is used only as the public comparison target through the existing helper.",
            "diagnostic_scope": "waveform impact only; no production coefficient change and no strict gate claim",
            "uncertainty": "N=61 is a metadata hypothesis inferred from published coefficient intervals, not a proven historical grid or production full-precision source.",
        },
        "choices": {
            "cases": "exactly the four CASES from diagnose_hm_qnm_leaver_clean.py",
            "frequency_source": "exactly diagnose_hm_qnm_leaver_clean.FREQ",
            "frequency_hz": [float(value) for value in helper.FREQ],
            "mode_labels": MODE_LABELS,
            "overtone": 0,
            "fit_variant": "MAIN only: exact N=61 full-precision coefficients; N=257 MAIN sensitivity; printed London control",
            "exact_N61_grid_hypothesis": "a=0.995*sin(theta), theta uniform on [-pi/2, pi/2], count=61; inferred from printed intervals only",
            "join_amplitude": JOIN_AMPLITUDE,
            "join_phase": JOIN_PHASE,
            "join_forwarding": join_metadata,
            "base_fr22": "per_case_D_carrier",
            "base_fd22": "per_case_D_carrier",
            "carrier": "black-box IMRPhenomD callable supplied to the fresh assembly",
            "qnm_scaling": "fit dimensionless omega is converted to total-mass Mf frequency and positive damping before assembly",
            "d_carrier_qnm_sampling": "independent evaluate_d_qnm_220(final_spin), with total_mass/final_mass scaling",
            "selection_rule": "no fit variant or join choice selected from public waveform residuals",
            "runtime_coefficients_changed": False,
        },
        "execution": {
            "host": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "command": command,
            "pid": os.getpid(),
            "log_path": str(args.log_path) if args.log_path else None,
            "next_check": "inspect progress.json during a redirected run; final report.json is written after all four cases",
            "stop_command": f"kill {os.getpid()}",
        },
        "fit_inputs": {"n61_main": exact_metadata, "n257_main": n257_metadata},
        "source_hashes_before": source_hashes_before,
        "versions": {},
        "cases": [],
        "status": "running",
    }
    try:
        import lal
        import lalsimulation
        import torch

        report["versions"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "lal": lal.__version__,
            "lalsimulation": lalsimulation.__version__,
        }
    except Exception as exc:
        report["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "error": str(exc)}
    _write_json(output_dir / "progress.json", report)

    for index, case in enumerate(helper.CASES, start=1):
        print(f"evaluating HM exact-N=61 case {index}/{len(helper.CASES)}", flush=True)
        case_started = time.monotonic()
        row = _case(case, fit_sets, join_kwargs)
        row["elapsed_seconds"] = time.monotonic() - case_started
        report["cases"].append(row)
        _write_json(output_dir / "progress.json", report)
        print(f"completed HM case {index}/{len(helper.CASES)} elapsed={row['elapsed_seconds']:.2f}s", flush=True)

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
