"""Signed-spin 220 frequencies from TorchWave's own Kerr equation solves.

No inherited frequency coefficients, external frequency tables or external
solver supply this runtime. See scripts/generate_kerr_qnm_independent.py and
our independent QNM report for numerical provenance and validation.
"""

from typing import Tuple

import torch

from torchwave.approximants.kerr_qnm_generated import signed_qnm_220


def evaluate_kerr_qnm_220(a_final: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Re(M_f*omega) and -Im(M_f*omega), preserving shape/device/dtype.

    Negative spin uses the positive-spin m=-2 branch by Kerr symmetry. Inputs
    outside [-0.9999, 0.9999] retain the historical endpoint-clamping behavior.
    """
    if not a_final.is_floating_point():
        raise TypeError("Kerr spin must be a floating-point tensor")
    working_dtype = torch.float32 if a_final.device.type == "mps" else torch.float64
    spin = a_final.to(working_dtype).clamp(min=-0.9999, max=0.9999)
    return tuple(value.to(a_final.dtype) for value in signed_qnm_220(spin))
