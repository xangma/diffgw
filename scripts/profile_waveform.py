#!/usr/bin/env python3
"""Convenience script wrapper for torchwave.profile.

Executes the TorchWave profiling and diagnostics suite.
Equivalent to running:
    python -m torchwave.profile ...
or using the CLI:
    torchwave-profile ...
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path when run directly from repo checkout
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from torchwave.profile import (  # noqa: F401
    main,
    run_cprofile,
    run_memory_profile,
    run_torch_profiler,
)

if __name__ == "__main__":
    main()
