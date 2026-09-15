"""Authoritative scalar policies for frozen TEOB trajectory replay."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def cache_fixed_step(data: Mapping, explicit: float | None = None) -> tuple[float, str]:
    """Read the saved integration step, never infer it from rounded timestamps.

    Older caches omit this field and require an explicit, independently known
    value. An explicit value cannot silently override authoritative metadata.
    """
    def scalar(value, name):
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in 'fiu':
            raise ValueError(f'{name} must be a positive finite scalar')
        result = float(array)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f'{name} must be a positive finite scalar')
        return result

    if explicit is not None:
        explicit = scalar(explicit, 'explicit fixed step')
    if 'continuation_fixed_step' in data:
        step = scalar(data['continuation_fixed_step'], 'cached fixed step')
        if explicit is not None and explicit != step:
            raise ValueError('explicit fixed step conflicts with cached continuation_fixed_step')
        provenance = 'cache.continuation_fixed_step'
    elif explicit is not None:
        step, provenance = explicit, 'explicit --fixed-step for legacy cache'
    else:
        raise ValueError('cache lacks continuation_fixed_step; provide independently known --fixed-step (do not infer it from timestamps)')
    if 'nqc_auxiliary_spacing' in data:
        auxiliary = scalar(data['nqc_auxiliary_spacing'], 'cached auxiliary spacing')
        if auxiliary != min(.5, step):
            raise ValueError('cached auxiliary spacing conflicts with the fixed-step policy')
    return step, provenance
