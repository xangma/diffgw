#!/usr/bin/env python3
"""Compatibility wrapper delegating to torchwave.qnm.generate."""

import sys
from pathlib import Path

# Ensure src is on path if running as a standalone script without installation
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from torchwave.qnm.generate import *
from torchwave.qnm.generate import main

if __name__ == "__main__":
    main()
