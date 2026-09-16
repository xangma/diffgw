"""Controls that prevent interpolation and step precision from invalidating replay."""
import numpy as np
import pytest
import torch

from scripts import teob_trajectory_grid_controls as replay


def test_prefix_keeps_exact_knots_and_uses_cubic_between_them():
    t = torch.tensor([0., 1., 2., 3., 4.], dtype=torch.float64)
    y = t**3 - 2*t
    query = torch.tensor([0., .5, 2., 3.5, 4.], dtype=t.dtype)
    actual = replay.resample_prefix(t, y, query)
    assert torch.equal(actual[[0, 2, 4]], y[[0, 2, 4]])
    # Independent spline implementation, including natural endpoint conditions.
    from scipy.interpolate import CubicSpline
    np.testing.assert_allclose(actual, CubicSpline(t, y, bc_type='natural')(query), rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match='extrapolation'):
        replay.resample_prefix(t, y, t - .01)


def test_explicit_step_clone_matches_native_on_its_own_auxiliary_grid():
    data = dict(np.load(replay.old.DEFAULT_CACHE, allow_pickle=False))
    with torch.no_grad():
        t, state, kwargs = replay.trajectory_inputs(data, 'tight_0p1')
        native = replay.nqc_module.assemble_teob_merger_nqc(t, state, **kwargs)
        # Large epoch makes adjacent floating-point subtraction differ from dt.
        assert float(native.auxiliary.time[1] - native.auxiliary.time[0]) != .1
        fixed = replay.explicit_spacing_nqc(t, state, **kwargs, auxiliary_time_override=native.auxiliary.time)
        assert float(fixed.auxiliary_spacing) == .1
        assert torch.equal(fixed.coefficients.amplitude, native.coefficients.amplitude)
        assert torch.equal(fixed.coefficients.phase, native.coefficients.phase)
        bad = native.auxiliary.time.clone()
        bad[10] += .001
        with pytest.raises(ValueError, match='authoritative step'):
            replay.explicit_spacing_nqc(t, state, **kwargs, auxiliary_time_override=bad)


def test_resampling_own_prefix_does_not_change_native_assembly():
    data = dict(np.load(replay.old.DEFAULT_CACHE, allow_pickle=False))
    with torch.no_grad():
        t, state, kwargs = replay.trajectory_inputs(data, 'tight_0p1')
        nqc = replay.nqc_module.assemble_teob_merger_nqc(t, state, **kwargs)
        native, _ = replay.assemble(t, nqc)
        controlled, _ = replay.assemble(t, nqc, prefix=t[t < nqc.auxiliary.time[0]])
        for key in ('joined_time', 'extended_time', 'time', 'amplitude', 'phase', 'mode'):
            assert torch.equal(getattr(native, key), getattr(controlled, key))
