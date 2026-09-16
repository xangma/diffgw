from __future__ import annotations

import numpy as np
import pytest

from scripts.teob_cache_metadata import cache_fixed_step


def test_large_timestamp_roundoff_cannot_replace_saved_step():
    times = 4_552_206.307415748 + np.arange(200)*.1
    assert times[1]-times[0] != .1
    cache = {'continuation_fixed_step': np.array(.1),
             'nqc_auxiliary_spacing': np.array(.1), 'nqc_auxiliary_time': times}
    assert cache_fixed_step(cache) == (.1, 'cache.continuation_fixed_step')
    with pytest.raises(ValueError, match='conflicts'):
        cache_fixed_step(cache, times[1]-times[0])


def test_legacy_cache_requires_explicit_independent_step():
    cache = {'nqc_auxiliary_time': np.arange(200)*.1}
    with pytest.raises(ValueError, match='independently known'):
        cache_fixed_step(cache)
    assert cache_fixed_step(cache, .1)[0] == .1


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), [.1], True])
def test_invalid_policy_is_rejected(value):
    with pytest.raises(ValueError, match='positive finite scalar'):
        cache_fixed_step({'continuation_fixed_step': value})


def test_auxiliary_cap_is_checked_without_changing_integration_step():
    assert cache_fixed_step({'continuation_fixed_step': 1., 'nqc_auxiliary_spacing': .5})[0] == 1.
    with pytest.raises(ValueError, match='auxiliary spacing conflicts'):
        cache_fixed_step({'continuation_fixed_step': .1, 'nqc_auxiliary_spacing': .2})
