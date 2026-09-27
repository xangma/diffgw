"""Source-derived mode-pair and gauge identities, independent of NR fitting."""

import math

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")
sp = pytest.importorskip("sympy")

from diffgw_reference import imrphenomhm_sxs_conventions as ref


def test_symbolic_relative_phase_cancels_time_and_orbit():
    m = sp.symbols("m", integer=True, positive=True)
    f, alpha, tau, base, gamma = sp.symbols(
        "f alpha tau B gamma", real=True,
    )
    phi_m = ref.symbolic_gauge_phase(m, f, alpha, tau, base) + gamma
    phi_22 = ref.symbolic_gauge_phase(2, 2 * f / m, alpha, tau, base) + gamma
    assert sp.simplify(2 * phi_m - m * phi_22
                       - (m - 2) * (base - gamma)) == 0
    assert sp.simplify(phi_m.subs({m: 2, base: 0})
                       - (2 * alpha - 2 * sp.pi * f * tau + gamma)) == 0
    with pytest.raises(ValueError, match="fourier_sign"):
        ref.symbolic_gauge_phase(m, f, alpha, tau, base, fourier_sign=0)


@pytest.mark.parametrize("fourier_sign", (-1, 1))
def test_sampled_fourier_shift_and_negative_m_pair(fourier_sign):
    # A complex chirp probes the +/-m conjugation and the frequency reversal.
    t = np.linspace(-8, 8, 32769)
    f, tau, alpha, m = 0.32, 0.17, 0.39, 3

    def positive_mode(times):
        return np.exp(-math.pi * (times ** 2) / 2) * np.exp(
            1j * (1.2 * times + 0.16 * times ** 2))

    def fourier(values, frequency):
        return np.trapezoid(
            values * np.exp(fourier_sign * 2j * math.pi * frequency * t), t,
        )
    bare = fourier(positive_mode(t), f)
    shifted = fourier(positive_mode(t - tau) * np.exp(1j * m * alpha), f)
    expected = bare * complex(ref.gauge_factor(
        m, f, orbital_phase=alpha, time_shift=tau,
        fourier_sign=fourier_sign,
    ))
    np.testing.assert_allclose(shifted, expected, rtol=3e-13, atol=3e-13)
    # The paper's physical rotation acts on h_lm; the public positive carrier
    # is its conjugate, so delta=-phic supplies the native +m*phic angle.
    physical_rotation = -alpha
    sxs_positive_carrier = fourier(np.conj(positive_mode(t)), f)
    rotated_sxs_carrier = fourier(
        np.conj(positive_mode(t) * np.exp(1j * m * physical_rotation)), f,
    )
    np.testing.assert_allclose(
        rotated_sxs_carrier,
        sxs_positive_carrier * np.exp(1j * m * alpha),
        rtol=3e-13, atol=3e-13,
    )
    for ell in (2, 3, 4):
        negative_mode = (-1) ** ell * np.conj(positive_mode(t))
        negative_m_spectrum = fourier(negative_mode, f)
        np.testing.assert_allclose(
            negative_m_spectrum,
            complex(ref.paired_spectrum(fourier(positive_mode(t), -f), ell)),
            rtol=3e-13, atol=3e-13,
        )
        np.testing.assert_allclose(
            fourier(np.conj(positive_mode(t)), f),
            complex(ref.positive_carrier_from_negative_m(
                negative_m_spectrum, ell)),
            rtol=3e-13, atol=3e-13,
        )


@pytest.mark.parametrize("m", (1, 2, 3, 4))
def test_numerical_relative_invariant_and_declared_sign(m):
    with mp.workdps(60):
        f, alpha, tau, base = map(mp.mpf, (".37", ".48", ".11", ".29"))
        mode = mp.mpc("1.2", "-.6")
        carrier = mp.mpc(".7", ".9")
        before = ref.doubled_relative_phase(mode, carrier, m)
        transformed = ref.doubled_relative_phase(
            mode * ref.gauge_factor(m, f, orbital_phase=alpha,
                                    time_shift=tau, reference_base=base,
                                    global_phase=mp.pi),
            carrier * ref.gauge_factor(2, 2 * f / m, orbital_phase=alpha,
                                       time_shift=tau, reference_base=base,
                                       global_phase=mp.pi),
            m,
        )
        predicted = before * mp.exp(mp.j * (m - 2) * base) * (-1) ** m
        assert abs(transformed - predicted) < mp.mpf("1e-55")
        assert abs(abs(transformed) - 1) < mp.mpf("1e-55")
    with pytest.raises(ValueError, match="zero mode"):
        ref.doubled_relative_phase(0, 1, m)
