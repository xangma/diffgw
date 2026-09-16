#!/usr/bin/env python3
"""Compatibility wrapper delegating to torchwave.qnm.solver."""

import sys
from pathlib import Path

# Ensure src is on path if running as a standalone script without installation
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from torchwave.qnm.solver import *
from torchwave.qnm.solver import (
    _angular_basis,
    _angular_extended,
    _radial_parameters,
    _fraction,
    _correct,
    _wkb_seed,
    _schwarzschild_roots,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kerr Leaver utilities")
    parser.add_argument("--angular-json", metavar="PATH", help="write independent angular convergence JSON")
    args = parser.parse_args()
    if args.angular_json:
        angular_verification_report(args.angular_json)
