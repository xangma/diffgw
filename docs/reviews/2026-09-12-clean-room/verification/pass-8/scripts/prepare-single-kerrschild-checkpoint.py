#!/usr/bin/env python3
"""Compatibility entry point for the corrected structural renderer."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if len(sys.argv) != 4:
    raise SystemExit("usage: TEMPLATE OUTPUT_DIR RECORD")
subprocess.run([
    "ruby", str(HERE / "prepare-single-kerrschild-checkpoint-corrected.rb"),
    *sys.argv[1:],
], check=True)
