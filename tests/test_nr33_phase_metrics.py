import numpy as np
import pytest

from scripts.nr33_phase_metrics import optimized_overlap, phase_metrics


def test_constant_and_time_shift_are_removed():
    f = np.linspace(.095, .3, 200)
    target = (1 + .2 * f) * np.exp(1j * (.3 + 2 * np.pi * f * .17))
    model = (1 + .2 * f) * np.exp(1j * .1)
    m = phase_metrics(f, model, target)
    assert m["raw_relative_l2"] > .1
    assert m["constant_relative_l2"] > .03
    assert m["affine_relative_l2"] < 1e-12
    assert m["affine_overlap"] > 1 - 1e-12
    assert m["affine_mismatch"] < 1e-12
    assert abs(m["time_shift"] - .17) < 1e-10


def test_nonlinear_phase_remains_after_affine_alignment():
    f = np.linspace(.095, .3, 300)
    nonlinear = .8 * (f - .2) ** 2
    target = np.exp(1j * nonlinear)
    model = np.ones_like(f, dtype=complex)
    m = phase_metrics(f, model, target)
    assert m["affine_relative_l2"] > 1e-3
    assert m["affine_phase_rms"] > 1e-3
    assert m["affine_overlap"] < 1


def test_weighting_floor_and_bounds():
    f = np.linspace(.1, .3, 100)
    target = np.exp(1j * 2 * np.pi * f * 2)
    model = np.ones_like(f, dtype=complex)
    m = phase_metrics(f, model, target, weights=np.linspace(1, 2, f.size), time_bounds=(0, 1))
    assert 0 <= m["time_shift"] <= 1
    with pytest.raises(ValueError):
        phase_metrics(f, model, target, time_bounds=(2, 1))


@pytest.mark.parametrize("bad", [np.zeros(5), np.full(5, np.nan)])
def test_rejects_invalid_support(bad):
    f = np.linspace(.1, .2, 5)
    with pytest.raises(ValueError):
        phase_metrics(f, np.ones(5, complex), bad)


def test_optimizer_recovers_100m_shift_against_dense_grid():
    f = np.linspace(.1, .3, 1000)
    amp = 0.2 + 4 * (f - .1) ** 2
    reference = amp * np.exp(1j * (0.4 + 2 * np.pi * f * 100.0))
    candidate = amp * np.exp(1j * .4)
    result = optimized_overlap(f, candidate, reference, time_bounds=(-200, 200))
    dense = np.linspace(-200, 200, 20001)
    scores = [abs(np.sum(np.conj(candidate) * reference * np.exp(-2j*np.pi*f*t)))
              for t in dense]
    assert abs(result["time_shift"] - 100) < .02
    assert result["overlap"] >= max(scores) / np.sqrt(np.sum(abs(candidate)**2)*np.sum(abs(reference)**2)) - 1e-6
    assert result["boundary"] is False


def test_optimizer_reports_boundary():
    f = np.linspace(.1, .3, 300)
    reference = np.exp(1j * 2*np.pi*f*20)
    result = optimized_overlap(f, np.ones_like(f, complex), reference, time_bounds=(0, 1))
    assert result["boundary"] is True
    assert result["time_shift"] in (0, 1) or np.isclose(result["time_shift"], 0, atol=1e-6)


def test_phase_mask_excludes_weak_noisy_tail_from_phase_only():
    f = np.linspace(.1, .3, 200)
    amp = np.ones(200); amp[-30:] = 1e-5
    reference = amp * np.exp(1j * .2)
    candidate = reference.copy()
    candidate[-30:] = 1.0 + 0j
    masked = phase_metrics(f, candidate, reference, phase_mask=amp > 1e-3)
    full = phase_metrics(f, candidate, reference)
    clean = phase_metrics(f, reference, reference, phase_mask=amp > 1e-3)
    assert masked["constant_phase"] == pytest.approx(0.0, abs=1e-5)
    assert masked["phase_mask_retained_reference_norm_fraction"] < 1
    assert full["raw_relative_l2"] > clean["raw_relative_l2"]


def test_phase_fit_uses_reference_amplitude_squared_and_scale_invariant_weights():
    f = np.linspace(.1, .3, 100)
    reference = (0.1 + 10*f) * np.exp(1j * (0.2 + 2*np.pi*f*.4 + 9*(f-.2)**2))
    candidate = np.abs(reference) * np.exp(1j*.2)
    w = np.linspace(1, 3, f.size)
    a = phase_metrics(f, candidate, reference, weights=w)
    b = phase_metrics(f, candidate, reference, weights=17*w)
    assert a["time_shift"] == pytest.approx(b["time_shift"], abs=1e-10)
    x = np.column_stack((np.ones(len(f)), 2*np.pi*f))
    y = np.unwrap(np.angle(reference * np.conj(candidate)))
    phasew = w * np.abs(reference)**2
    expected = np.linalg.lstsq(x * np.sqrt(phasew[:, None]), y * np.sqrt(phasew), rcond=None)[0]
    assert a["time_shift"] == pytest.approx(expected[1], abs=1e-10)
    assert a["affine_phase"] == pytest.approx(expected[0], abs=1e-10)
    assert a["phase_mask_retained_reference_norm_fraction"] == pytest.approx(1)


def test_phase_rms_uses_mask_and_reference_amplitude_squared():
    f = np.linspace(.1, .3, 100)
    amp = np.ones(100); amp[:50] = 10
    reference = amp.astype(complex)
    candidate = reference * np.exp(-1j * np.linspace(.02, .3, 100))
    mask = np.arange(100) >= 20
    result = phase_metrics(f, candidate, reference, phase_mask=mask)
    expected = np.sqrt(np.average(np.linspace(.02, .3, 100)[mask]**2, weights=amp[mask]**2))
    assert result["raw_phase_rms"] == pytest.approx(expected, abs=1e-10)
    assert result["phase_mask_retained_reference_norm_fraction"] < .7


def test_single_nan_is_rejected_even_outside_phase_mask():
    f = np.linspace(.1, .3, 20)
    candidate = np.ones(20, complex); reference = np.ones(20, complex)
    candidate[0] = np.nan
    with pytest.raises(ValueError):
        phase_metrics(f, candidate, reference, phase_mask=np.arange(20) > 0)
