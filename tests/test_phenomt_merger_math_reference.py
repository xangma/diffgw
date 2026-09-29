"""Paper-level supplied-input merger checks for both dominant-mode carriers.

The boundary values and QNM rates are calibrated inputs.  This gate tests the
Eq. (28)-(32) interpolation and joins, not their fitted surfaces or NR truth.
"""

import importlib

import mpmath as mp
import numpy as np
import pytest

from diffgw_reference import phenomt_merger as reference


CASES = (
    (30.0, 30.0, 0.0, 0.0),
    (35.0, 25.0, 0.7, -0.5),
    (60.0, 6.0, 0.9, 0.1),
)


def _native(backend):
    if backend == "torchwave":
        import torch

        def array(value):
            return torch.as_tensor(value, dtype=torch.float64)
    else:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        def array(value):
            return jnp.asarray(value, dtype=jnp.float64)

    return importlib.import_module(backend).IMRPhenomT(), array


def _scalar(value):
    return float(np.asarray(value))


def test_symbolic_paper_forms_match_scalar_values_and_slopes():
    import sympy as sp

    (t, alpha, peak_time, wr), a_symbols, b_symbols, omega, amplitude = reference.symbolic_ansatz()
    a = ("0.1", "0.2", "-0.03", "0.004", "0.005")
    b = ("0.1", "0.002", "0.03", "0.04")
    substitutions = {t: -7, alpha: sp.Rational(9, 100), peak_time: 0,
                     wr: sp.Rational(55, 100)}
    substitutions.update({symbol: sp.Rational(value) for symbol, value in zip(a_symbols, a)})
    substitutions.update({symbol: sp.Rational(value) for symbol, value in zip(b_symbols, b)})
    for expression, expected in (
        (omega, reference.frequency(-7, "0.55", "0.09", tuple(map(mp.mpf, a)))),
        (sp.diff(omega, t), reference.frequency_slope(-7, "0.55", "0.09", tuple(map(mp.mpf, a)))),
        (amplitude, reference.amplitude(-7, "0.09", 0, tuple(map(mp.mpf, b)))),
        (sp.diff(amplitude, t), reference.amplitude_slope(-7, "0.09", 0, tuple(map(mp.mpf, b)))),
    ):
        np.testing.assert_allclose(float(expression.evalf(60, subs=substitutions)), float(expected), rtol=2e-15)


def test_scalar_merger_collocation_and_derivatives():
    with mp.workdps(75):
        frequency_input = (-140, -30, 0.55, 0.09, 0.08, 0.0005, 0.2, 0.4, 0.004)
        a = reference.frequency_coefficients(*frequency_input)
        cut, mid, wr, alpha, wc, dc, wm, wp, dp = frequency_input
        for t, target in ((cut, wc), (mid, wm), (0, wp)):
            assert mp.almosteq(reference.frequency(t, wr, alpha, a), mp.mpf(str(target)))
        for t, target in ((cut, dc), (0, dp)):
            assert mp.almosteq(reference.frequency_slope(t, wr, alpha, a), mp.mpf(str(target)))
        for t in (-100, -40, -7):
            step = mp.mpf("1e-25")
            assert mp.almosteq(
                reference.frequency_slope(t, wr, alpha, a),
                (reference.frequency(t + step, wr, alpha, a)
                 - reference.frequency(t - step, wr, alpha, a)) / (2 * step),
                rel_eps=mp.mpf("1e-45"),
            )
            assert mp.almosteq(
                (reference.phase_increment(t + step, cut, wr, alpha, a)
                 - reference.phase_increment(t - step, cut, wr, alpha, a)) / (2 * step),
                reference.frequency(t, wr, alpha, a),
                rel_eps=mp.mpf("1e-45"),
            )

        amplitude_input = (-150, -25, 0, 0.09, 0.02, 0.0001, 0.15, 0.3)
        b = reference.amplitude_coefficients(*amplitude_input)
        ac, am, peak, rate, hc, dc, hm, hp = amplitude_input
        for t, target in ((ac, hc), (am, hm), (peak, hp)):
            assert mp.almosteq(reference.amplitude(t, rate, peak, b), mp.mpf(str(target)))
        assert mp.almosteq(reference.amplitude_slope(ac, rate, peak, b), mp.mpf(str(dc)))
        for t in (-100, -40, -7):
            assert mp.almosteq(
                reference.amplitude_slope(t, rate, peak, b),
                (reference.amplitude(t + step, rate, peak, b)
                 - reference.amplitude(t - step, rate, peak, b)) / (2 * step),
                rel_eps=mp.mpf("1e-45"),
            )


@pytest.mark.parametrize("backend", ("torchwave", "jaxwave"))
@pytest.mark.parametrize("case", CASES, ids=lambda x: "-".join(map(str, x)))
def test_production_merger_frequency_phase_amplitude(backend, case):
    model, array = _native(backend)
    native_module = importlib.import_module(f"{backend}.approximants.imrphenomt")
    amp_module = importlib.import_module(f"{backend}.approximants._phenomt_amplitude")
    state = model._coeff(*map(array, case))
    eta, delta, chi1, chi2, wc = state[1], state[2], state[5], state[6], state[7]
    cut = _scalar(state[25])
    mid = -5 / (_scalar(eta) * 0.95**8)
    wr, alpha = _scalar(state[21]), _scalar(state[20])
    omega_cut = _scalar(model._omega_insp(array([cut]), eta, delta, chi1, chi2, wc)[0])
    slope_cut = _scalar(native_module._domega_insp(array([cut]), eta, chi1, chi2, delta, wc)[0])
    omega_mid, omega_peak = _scalar(state[16]), _scalar(state[13])
    slope_peak = -wr * _scalar(state[23])
    a = reference.frequency_coefficients(
        cut, mid, wr, alpha, omega_cut, slope_cut, omega_mid, omega_peak, slope_peak
    )
    native_a = [1 - omega_peak / wr, _scalar(state[23]) / alpha]
    native_a.extend(np.asarray(state[24]).tolist())
    np.testing.assert_allclose(native_a, [float(c) for c in a], rtol=2e-10, atol=2e-11)

    times = (cut, cut * 0.85, mid, -25.0, -7.0, 0.0)
    u = array(times)
    native_omega, native_phase, amp_state = model._evaluate(u, state)
    native_slope = model._domega_full(u, state)
    expected_omega = [float(reference.frequency(t, wr, alpha, a)) for t in times]
    expected_slope = [float(reference.frequency_slope(t, wr, alpha, a)) for t in times]
    expected_phase = [float(reference.phase_increment(t, cut, wr, alpha, a)) for t in times]
    np.testing.assert_allclose(np.asarray(native_omega), expected_omega, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(np.asarray(native_slope), expected_slope, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(
        np.asarray(native_phase) - _scalar(native_phase[0]), expected_phase, rtol=2e-10, atol=2e-9
    )

    # The PN cutoff value and slope are supplied by the existing carrier.
    # The paper's separate four-condition amplitude solve is independently
    # reconstructed below, with its own basis ordering.
    amp_alpha = _scalar(amp_state[3])
    w_amp_cut = _scalar(model._evaluate(array([-150.0]), state)[0][0])
    dw_amp_cut = _scalar(model._domega_full(array([-150.0]), state)[0])
    x_cut = (w_amp_cut / 2) ** (2 / 3)
    amp_at_cut, da_dx = amp_module._pn(array([x_cut]), amp_state[0], amp_state[1], derivative=True)
    amp_at_cut, da_dx = complex(np.asarray(amp_at_cut[0])), complex(np.asarray(da_dx[0]))
    signed_cut = np.sign(amp_at_cut.real) * abs(amp_at_cut)
    signed_slope = (
        np.sign(amp_at_cut.real) * (amp_at_cut.conjugate() * da_dx).real / abs(amp_at_cut)
        * (2 / 3) * x_cut / w_amp_cut * dw_amp_cut
    )
    b = reference.amplitude_coefficients(
        -150, -25, 0, amp_alpha, signed_cut, signed_slope, _scalar(state[10]), _scalar(state[9])
    )
    native_b = np.asarray(amp_state[2])[[0, 3, 2, 1]]
    np.testing.assert_allclose(native_b, [float(c) for c in b], rtol=2e-10, atol=2e-11)
    amp_times = (-150.0, -120.0, -70.0, -25.0, -5.0, 0.0)
    native_amp = amp_module.amplitude(
        array(amp_times), amp_state, lambda t: model._evaluate(t, state)[0]
    )
    expected_amp = [abs(float(reference.amplitude(t, amp_alpha, 0, b))) for t in amp_times]
    np.testing.assert_allclose(np.asarray(native_amp), expected_amp, rtol=2e-10, atol=2e-11)
