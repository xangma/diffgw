#!/usr/bin/env python3
"""Run the reviewed public TEOB replay with an opt-in closed potential.

This diagnostic delegates all replay inputs, policy arguments, native/public
comparisons, and cache handling to
``replay_teob_conditioned_waveform_public_clean.py``.  During that call only
``teobresums._teobresums_orbital_potential`` is temporarily replaced by the
clean closed-form P^1_5 helper.  The original symbol is restored in a
``finally`` block.  Default outputs are kept in a separate verification
directory so this experiment cannot overwrite the public replay.

All options accepted by the delegated replay remain available, including
``--pa-initializer-policy analytic`` and ``--pa-initializer-policy
direct-quadratic-value``.  This script does not select or change those
policies.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import torchwave.approximants.teobresums as teobresums
from torchwave.approximants.eob.teob_orbital_potential_closed_clean import (
    teobresums_orbital_potential_closed,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-closed-potential-replay-clean"
DEFAULT_REPORT = VERIFY / "replay-report.json"
DEFAULT_NATIVE = VERIFY / "replay-native-output.npz"
DEFAULT_FD = VERIFY / "replay-public-fd-output.npz"
DEFAULT_LOG = VERIFY / "replay.log"
HELPER = ROOT / "src/torchwave/approximants/eob/teob_orbital_potential_closed_clean.py"
WRAPPER = Path(__file__).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_option(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in argv)


def _option_path(argv: list[str], name: str, default: Path) -> Path:
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires a path")
            return Path(argv[index + 1])
        prefix = name + "="
        if arg.startswith(prefix):
            return Path(arg[len(prefix):])
    return default


def _delegated_argv(argv: list[str]) -> list[str]:
    """Add isolated defaults while preserving every caller supplied option."""

    result = list(argv)
    defaults = (
        ("--report", DEFAULT_REPORT),
        ("--native-output", DEFAULT_NATIVE),
        ("--fd-output", DEFAULT_FD),
        ("--log-path", DEFAULT_LOG),
    )
    for name, path in defaults:
        if not _has_option(result, name):
            result.extend((name, str(path)))
    return result


def _add_hash_record(report_path: Path, before: dict[str, str], after: dict[str, str],
                     restored: bool, original_command: str) -> None:
    """Attach substitution provenance to the delegated report."""

    if not report_path.is_file():
        return
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    report["closed_potential_substitution"] = {
        "patched_symbol": "torchwave.approximants.teobresums._teobresums_orbital_potential",
        "replacement": "torchwave.approximants.eob.teob_orbital_potential_closed_clean.teobresums_orbital_potential_closed",
        "scope": "temporary single-symbol substitution during delegated replay",
        "restored_in_finally": restored,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_hashes_unchanged": before == after,
        "wrapper_command": original_command,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fresh_delegated_report(report_path: Path, previous_sha256: str | None,
                            invocation_started: float) -> bool:
    """Require evidence that the delegated call wrote this report."""

    if not report_path.is_file():
        return False
    if previous_sha256 is not None and _sha256(report_path) == previous_sha256:
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run_control = report["run_control"]
        started = float(run_control["started_unix"])
        return int(run_control["pid"]) == os.getpid() and started >= invocation_started
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    original_argv = list(sys.argv)
    caller_args = original_argv[1:]
    delegated_args = _delegated_argv(caller_args)
    report_path = _option_path(delegated_args, "--report", DEFAULT_REPORT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous_report_sha256 = _sha256(report_path) if report_path.is_file() else None
    invocation_started = time.time()
    source_before = {"wrapper": _sha256(WRAPPER), "closed_potential_helper": _sha256(HELPER)}
    original_potential = teobresums._teobresums_orbital_potential
    restored = False
    try:
        # Importing the existing script keeps its public argument parser and
        # comparison code as the sole replay implementation.
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import replay_teob_conditioned_waveform_public_clean as delegated  # noqa: PLC0415

        teobresums._teobresums_orbital_potential = teobresums_orbital_potential_closed
        sys.argv = [str(Path(__file__).resolve()), *delegated_args]
        return delegated.main()
    finally:
        teobresums._teobresums_orbital_potential = original_potential
        restored = teobresums._teobresums_orbital_potential is original_potential
        sys.argv = original_argv
        source_after = {"wrapper": _sha256(WRAPPER), "closed_potential_helper": _sha256(HELPER)}
        if _fresh_delegated_report(report_path, previous_report_sha256, invocation_started):
            _add_hash_record(report_path, source_before, source_after, restored,
                             shlex.join([sys.executable, *original_argv]))


if __name__ == "__main__":
    raise SystemExit(main())
