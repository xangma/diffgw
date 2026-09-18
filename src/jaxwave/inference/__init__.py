"""Gravitational-wave parameter inference and differentiable likelihood module in JAX.

Strict Clean-Room Implementation based exclusively on published literature:
- Cutler & Flanagan (1994), Phys. Rev. D 49, 2658 (arXiv:gr-qc/9402014)
- Veitch et al. (2015), Phys. Rev. D 91, 042003 (arXiv:1409.7788)
- Thrane & Talbot (2019), PASA 36, e010 (arXiv:1809.02293)
- Zackay, Dai, Venumhav (2018), arXiv:1806.08792
- Cornish (2010), arXiv:1007.4820
- Vallisneri (2008), Phys. Rev. D 77, 042001 (arXiv:gr-qc/0703086)
"""

import jax

# Automatically enable float64 in JAX for gravitational-wave phase and likelihood precision
jax.config.update("jax_enable_x64", True)

from jaxwave.inference.likelihood import (
    noise_weighted_inner_product,
    log_bessel_i0,
    optimal_snr,
    matched_filter_snr,
    WhittleLikelihood,
)
from jaxwave.inference.fisher import (
    fisher_information_matrix,
    covariance_matrix,
    sky_localization_area,
    optimize_map,
)
from jaxwave.inference.relative_binning import (
    RelativeBinningLikelihood,
    compute_relative_binning_bins,
    compute_relative_binning_summary,
)
from jaxwave.inference.bucketing import (
    get_bucketed_batch_size,
    pad_parameters_to_bucket,
    unpad_waveform_from_bucket,
    BucketedWaveformEvaluator,
)

__all__ = [
    "noise_weighted_inner_product",
    "log_bessel_i0",
    "optimal_snr",
    "matched_filter_snr",
    "WhittleLikelihood",
    "fisher_information_matrix",
    "covariance_matrix",
    "sky_localization_area",
    "optimize_map",
    "RelativeBinningLikelihood",
    "compute_relative_binning_bins",
    "compute_relative_binning_summary",
    "get_bucketed_batch_size",
    "pad_parameters_to_bucket",
    "unpad_waveform_from_bucket",
    "BucketedWaveformEvaluator",
]

