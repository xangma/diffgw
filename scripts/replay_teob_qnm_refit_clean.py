#!/usr/bin/env python3
"""Run the clean TEOB replay with a temporary QNM provider.

The JSON file is deliberately local and frozen.  It must contain a
``domain: [0.0, 0.95]`` and ``fits.{omega220,alpha220,alpha21}.coefficients``
arrays in the seven-parameter cubic-over-cubic rational basis.
Only ``teob_postmerger.qnm_fits`` is replaced during the delegated replay;
the original symbol is restored even when the replay raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import torch

from torchwave.approximants.eob import teob_postmerger
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-qnm-recalibration-clean/refit"
DEFAULT_REPORT = VERIFY / "replay-report.json"
DEFAULT_NATIVE = VERIFY / "replay-native-output.npz"
DEFAULT_FD = VERIFY / "replay-public-fd-output.npz"
DEFAULT_LOG = VERIFY / "replay.log"
DEFAULT_COEFFICIENTS = VERIFY.parent / "refit-calibration.json"
WRAPPER = Path(__file__).resolve()
POSTMERGER = ROOT / "src/torchwave/approximants/eob/teob_postmerger.py"
RINGDOWN = ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _option_path(argv: list[str], name: str, default: Path) -> Path:
    for i, arg in enumerate(argv):
        if arg == name:
            if i + 1 >= len(argv):
                raise ValueError(f"{name} requires a path")
            return Path(argv[i + 1])
        if arg.startswith(name + "="):
            return Path(arg[len(name) + 1:])
    return default


def _has_option(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in argv)


def _remove_option(argv: list[str], name: str) -> list[str]:
    """Remove this wrapper-only option before calling the delegated parser."""
    result: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == name:
            i += 2
        elif argv[i].startswith(name + "="):
            i += 1
        else:
            result.append(argv[i])
            i += 1
    return result


SPIN_MAX = 0.95


def _load_evaluator(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"QNM coefficient file does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("QNM coefficient file must contain a JSON object")
    if raw.get("domain") != [0.0, SPIN_MAX]:
        raise ValueError("QNM coefficient domain must be exactly [0.0, 0.95]")
    table = raw.get("fits")
    if not isinstance(table, dict):
        raise ValueError("QNM coefficient file needs a fits mapping")
    vectors: dict[str, tuple[float, ...]] = {}
    names = {"omega": "omega220", "alpha": "alpha220", "alpha21": "alpha21"}
    for name, key in names.items():
        entry = table.get(key)
        if isinstance(entry, dict):
            entry = entry.get("coefficients")
        if not isinstance(entry, list) or len(entry) != 7:
            raise ValueError(f"missing seven rational coefficients for {key}")
        try:
            values = [float(x) for x in entry]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid rational coefficients for {name}") from exc
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"non-finite rational coefficients for {name}")
        # Keep validation in agreement with the fitting experiment, including
        # continuous numerator/denominator pole checks.
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from refit_teob_qnm_independent_clean import validate_coefficients  # noqa: PLC0415
        validate_coefficients(values)
        vectors[name] = tuple(values)

    def evaluate(chi_final: torch.Tensor | float):
        if torch.as_tensor(chi_final).is_complex():
            raise ValueError("final spin must be real")
        chi = torch.as_tensor(chi_final)
        if not chi.is_floating_point():
            chi = chi.to(torch.get_default_dtype())
        if chi.is_complex() or torch.any(~torch.isfinite(chi)) or torch.any((chi < 0) | (chi > SPIN_MAX)):
            raise ValueError("final spin must be finite and satisfy 0 <= chi_final <= 0.95")
        result = []
        for name in ("omega", "alpha", "alpha21"):
            y0, b1, b2, b3, c1, c2, c3 = (chi.new_tensor(x) for x in vectors[name])
            value = y0 * (1 + b1 * chi + b2 * chi.square() + b3 * chi.pow(3))
            value = value / (1 + c1 * chi + c2 * chi.square() + c3 * chi.pow(3))
            result.append(value)
        return tuple(result)

    return evaluate, raw


def _direct_evaluator(spin: torch.Tensor | float):
    if torch.as_tensor(spin).is_complex():
        raise ValueError("final spin must be real")
    a = torch.as_tensor(spin)
    if not a.is_floating_point():
        a = a.to(torch.get_default_dtype())
    if torch.any(~torch.isfinite(a)) or torch.any((a < 0) | (a > SPIN_MAX)):
        raise ValueError("final spin must be finite and satisfy 0 <= chi_final <= 0.95")
    w0, al0 = dimensionless_qnm(2, 2, 0, a)
    _, al1 = dimensionless_qnm(2, 2, 1, a)
    return w0, al0, al1 - al0


def _delegated_argv(argv: list[str], output_root: Path = VERIFY) -> list[str]:
    result = list(argv)
    for name, path in (("--report", output_root / "replay-report.json"),
                       ("--native-output", output_root / "replay-native-output.npz"),
                       ("--fd-output", output_root / "replay-public-fd-output.npz"),
                       ("--log-path", output_root / "replay.log")):
        if not _has_option(result, name):
            result.extend((name, str(path)))
    return result


def _fresh_report(path: Path, previous: str | None, started: float) -> bool:
    if not path.is_file() or (previous is not None and _sha256(path) == previous):
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        control = data["run_control"]
        return int(control["pid"]) == os.getpid() and float(control["started_unix"]) >= started
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _record(path: Path, metadata: dict[str, Any]) -> None:
    if not path.is_file():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    report["qnm_refit_substitution"] = metadata
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    original_argv = list(sys.argv)
    caller = original_argv[1:]
    # Help remains available without requiring a coefficient artifact.
    if "--help" in caller or "-h" in caller:
        coeff_path = DEFAULT_COEFFICIENTS
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--qnm-coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
        parser.add_argument("--qnm-provider", choices=("refit", "direct"), default="refit")
        known, _ = parser.parse_known_args(caller)
        coeff_path = known.qnm_coefficients
        provider = known.qnm_provider
        if provider == "direct" and _has_option(caller, "--qnm-coefficients"):
            parser.error("--qnm-coefficients cannot be used with --qnm-provider direct")
    if "--help" in caller or "-h" in caller:
        provider = "refit"
    delegated = _remove_option(_remove_option(caller, "--qnm-coefficients"), "--qnm-provider")
    output_root = VERIFY.parent / "direct" if provider == "direct" else VERIFY
    delegated = _delegated_argv(delegated, output_root)
    if "--help" in caller or "-h" in caller:
        wrapper_parser = argparse.ArgumentParser(description=__doc__)
        wrapper_parser.add_argument("--qnm-provider", choices=("refit", "direct"), default="refit")
        wrapper_parser.add_argument("--qnm-coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
        wrapper_parser.print_help()
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import replay_teob_conditioned_waveform_public_clean as delegated_module  # noqa: PLC0415
        sys.argv = [str(WRAPPER), *delegated]
        try:
            return delegated_module.main()
        finally:
            sys.argv = original_argv
    report_path = _option_path(delegated, "--report", DEFAULT_REPORT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous = _sha256(report_path) if report_path.is_file() else None
    started = time.time()
    source_before = {"wrapper": _sha256(WRAPPER), "postmerger": _sha256(POSTMERGER),
                     "ringdown": _sha256(RINGDOWN),
                     "kerr_provider": _sha256(ROOT / "src/torchwave/approximants/kerr_qnm_generated.py"),
                     "fit_provider": _sha256(ROOT / "scripts/refit_teob_qnm_independent_clean.py"),
                     "coefficients": _sha256(coeff_path) if coeff_path.is_file() else "missing"}
    original = teob_postmerger.qnm_fits
    restored = False
    try:
        if provider == "direct":
            evaluator, raw = _direct_evaluator, {"provider": "kerr_qnm_generated.dimensionless_qnm"}
        else:
            evaluator, raw = _load_evaluator(coeff_path)
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import replay_teob_conditioned_waveform_public_clean as delegated_module  # noqa: PLC0415
        teob_postmerger.qnm_fits = evaluator
        sys.argv = [str(WRAPPER), *delegated]
        return delegated_module.main()
    finally:
        teob_postmerger.qnm_fits = original
        restored = teob_postmerger.qnm_fits is original
        sys.argv = original_argv
        source_after = {"wrapper": _sha256(WRAPPER), "postmerger": _sha256(POSTMERGER),
                        "ringdown": _sha256(RINGDOWN),
                        "kerr_provider": _sha256(ROOT / "src/torchwave/approximants/kerr_qnm_generated.py"),
                        "fit_provider": _sha256(ROOT / "scripts/refit_teob_qnm_independent_clean.py"),
                        "coefficients": _sha256(coeff_path) if coeff_path.is_file() else "missing"}
        if _fresh_report(report_path, previous, started):
            _record(report_path, {
                "patched_symbol": "torchwave.approximants.eob.teob_postmerger.qnm_fits",
                "provider": provider,
                "replacement": "JSON cubic-over-cubic rational evaluator" if provider == "refit" else "independent dimensionless Kerr QNM evaluator",
                "provider_call_site": "teob_postmerger.postmerger_22; used by teob_ringdown_assembly_clean",
                "scope": "temporary single-symbol substitution during delegated replay",
                "coefficient_path": str(coeff_path.resolve()) if provider == "refit" else None,
                "coefficient_sha256": source_before["coefficients"] if provider == "refit" else None,
                "coefficients": raw if provider == "refit" and 'raw' in locals() else None,
                "restored_in_finally": restored, "source_sha256_before": source_before,
                "source_sha256_after": source_after, "source_hashes_unchanged": source_before == source_after,
                "wrapper_command": shlex.join([sys.executable, *original_argv]),
            })


if __name__ == "__main__":
    raise SystemExit(main())
