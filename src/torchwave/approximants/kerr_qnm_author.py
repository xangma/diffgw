"""Compatibility import for the former author-data QNM backend.

Runtime frequencies now come exclusively from TorchWave's own Kerr equation
solves. This module retains no author frequency arrays. Historical external
fixtures remain in tests/data solely for verification; see THIRD_PARTY_NOTICES.md.
"""

from torchwave.approximants.kerr_qnm_generated import (
    SPIN_MIN, SPIN_MAX, SUPPORTED_MODES, dimensionless_qnm,
)

__all__ = ["SPIN_MIN", "SPIN_MAX", "SUPPORTED_MODES", "dimensionless_qnm"]
