"""Check the fixed-chart diagnostic on controlled sampled waveforms."""

import numpy as np

from dev.native_rom.diagnose_source_q4_fixed_anchor import fixed_anchor_metrics


def test_fixed_anchor_exposes_grid_dependent_coordinate_motion():
    fine_t = np.arange(-10., 10.01, .01)
    coarse_t = fine_t[::10]
    def model(t):
        return np.exp(-.03*t*t) * np.exp(1j*(.4*t + .01*t*t))
    anchor = dict(time_s=0., phase_rad=0.)
    same = fixed_anchor_metrics(coarse_t, model(coarse_t), fine_t,
                                model(fine_t), 1., anchor, anchor)
    shifted = fixed_anchor_metrics(coarse_t+.1, model(coarse_t), fine_t,
                                   model(fine_t), 1., anchor, anchor)
    nested = fixed_anchor_metrics(fine_t, model(fine_t), coarse_t,
                                  model(coarse_t), 1., anchor, anchor)
    assert same["metrics"]["complex_relative_l2"] < .001
    assert shifted["metrics"]["complex_relative_l2"] > .03
    assert nested["metrics"]["complex_relative_l2"] < 1e-12
