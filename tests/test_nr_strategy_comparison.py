import numpy as np
import pytest

from scripts.run_nr_strategy_comparison import canonical_modes, frequency_and_factor, scores


def synthetic_modes():
    t = np.linspace(-10, 10, 401)
    envelope = np.exp(-t**2 / 20)
    phase = .23 * t + .002 * t**2
    return t, envelope * np.exp(-2j * phase), .3j * envelope * np.exp(-3j * phase)


@pytest.mark.parametrize('rotation', [-4.2, -.2, 1.8, 3.5])
def test_canonical_gauge_is_invariant_under_one_physical_orbital_rotation(rotation):
    t, h22, h33 = synthetic_modes()
    original, peak, _ = canonical_modes(t, h22, h33)
    rotated, shifted_peak, _ = canonical_modes(t + 100,
        h22 * np.exp(2j * rotation), h33 * np.exp(3j * rotation))
    np.testing.assert_allclose(rotated, original, atol=2e-15)
    assert shifted_peak - peak == pytest.approx(100)


def test_relative_mode_phase_error_cannot_be_hidden_by_independent_alignment():
    _, h22, h33 = synthetic_modes()
    reference = np.stack((h22, h33))
    candidate = reference.copy()
    candidate[1] *= np.exp(.2j)
    result = scores(candidate, reference)
    assert result['modes']['33']['amplitude_l2'] < 1e-15
    assert result['modes']['33']['complex_l2'] == pytest.approx(2 * np.sin(.1))
    assert result['relative_mode_phase_rms_rad'] == pytest.approx(.2)
    assert result['coherent_equal_weight_sum_l2'] > 0


def test_odd_mode_scaling_and_declared_frequency_support():
    q = 2.5
    f, factor, a0 = frequency_and_factor(q, {'f_ring_33': .12, 'f_damp_33': .015})
    assert f.shape == (512,)
    assert f[0] == pytest.approx(.95 * .12)
    assert np.all(np.diff(f) > 0) and f[-1] < .3
    np.testing.assert_allclose(factor[1] / factor[0], (q - 1) / (q + 1))
    assert np.all(a0 > 0)
