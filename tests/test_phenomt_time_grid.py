"""Physical-time invariants for the continuous native precession prescription."""

import importlib

import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["torchwave", "jaxwave"])
@pytest.mark.parametrize("name", ["IMRPhenomTP", "IMRPhenomTPHM"])
def test_precession_is_independent_of_sampling_and_time_origin(backend, name):
    model = getattr(importlib.import_module(backend), name)()
    times = np.array([-0.12, -0.01, -0.001, 0.0, 0.001, 0.015])
    # Extra samples straddle merger, where a sample-index anchor would move.
    dense = np.sort(np.r_[times, [-0.003, -0.0001, 0.0002, 0.007]])
    indices = np.searchsorted(dense, times)
    kwargs = dict(mass1=60.0, mass2=6.0, chi1=0.8, chi2=0.1,
                  spin1x=0.2, spin1y=0.1, spin2x=0.1,
                  inclination=1.2, phic=0.1, f_ref=30.0)
    reference = model.time_domain(times, **kwargs)
    resampled = model.time_domain(dense, **kwargs)
    translated = model.time_domain(times + 0.25, tc=0.25, **kwargs)
    for ref, sampled, shifted in zip(reference, resampled, translated):
        ref = np.asarray(ref)
        norm = np.linalg.norm(ref)
        assert norm > 0
        assert np.linalg.norm(np.asarray(sampled)[indices] - ref) / norm < 1e-11
        assert np.linalg.norm(np.asarray(shifted) - ref) / norm < 1e-11
