#!/usr/bin/env python3
"""Convenience script wrapper for torchwave.benchmark.

Executes the TorchWave benchmarking suite.
Equivalent to running:
    python -m torchwave.benchmark ...
or using the CLI:
    torchwave-bench ...
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path when run directly from repo checkout
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from torchwave.benchmark import (  # noqa: F401
    ALL_MODELS,
    DEFAULT_MODEL_SPECS,
    POPULAR_MODELS,
    _has_lal,
    benchmark_lal_model,
    benchmark_torchwave_model,
    format_markdown_table,
    format_table,
    main,
    resolve_device,
    resolve_dtype,
    run_benchmark_suite,
    sync_device,
)

if __name__ == "__main__":
    main()
