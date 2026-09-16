#!/usr/bin/env python3
"""Compatibility command for TorchWave's independently derived Kerr generator."""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from torchwave.qnm.generate import main


if __name__ == "__main__":
    main()
