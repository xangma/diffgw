import math
import numpy as np
import pytest

from kerr_public_oracle import _simpson_fft, public_ringdown_fd, public_ringdown_fd_converged


def test_simpson_fft_polynomial_integral_at_zero_frequency():
    fs = 128.
    t = np.arange(101) / fs
    y = t * t
    out = _simpson_fft(y, 1. / fs, 256)
    assert np.isclose(out[0].real, (t[-1] ** 3) / 3., rtol=0, atol=2e-12)
    assert out[0].imag == 0


def test_simpson_fft_rejects_bad_inputs():
    with pytest.raises(ValueError):
        _simpson_fft(np.ones(3), 0., 8)
    with pytest.raises(ValueError):
        _simpson_fft(np.ones(9), 1., 8)
    with pytest.raises(ValueError):
        _simpson_fft(np.ones(9), 1., 0)
    with pytest.raises(ValueError):
        _simpson_fft(np.ones(9), 1., 8.0)


def test_simpson_fft_damped_cosine_nonzero_bins():
    fs, n = 256., 257
    t = np.arange(n) / fs
    decay, carrier = 3.2, 17.0
    out = _simpson_fft(np.exp(-decay * t) * np.cos(2 * np.pi * carrier * t), 1 / fs, 1024)
    T = t[-1]
    for k in (3, 17, 61):
        f = k * fs / 1024
        def integ(s):
            return (1 - np.exp(-s * T)) / s
        expected = .5 * (integ(decay + 2j * np.pi * (carrier + f)) +
                          integ(decay + 2j * np.pi * (f - carrier)))
        assert np.isclose(out[k], expected, rtol=0, atol=1e-5)


def test_public_adapter_optional_and_support():
    pytest.importorskip("lalsimulation")
    f, hp, hc = public_ringdown_fd(65., .7, 100., sample_rate=65536,
                                   delta_f=.25, f_max=500.)
    assert f[0] == 0 and f[-1] == 500.
    assert hp.shape == hc.shape == f.shape
    assert np.isfinite(hp).all() and np.isfinite(hc).all()


def test_public_adapter_sampling_contract_and_convergence():
    pytest.importorskip("lalsimulation")
    with pytest.raises(ValueError):
        public_ringdown_fd(65., .7, 100., sample_rate=1000.5)
    with pytest.raises(ValueError):
        public_ringdown_fd(65., 1., 100.)
    f, hp, hc, meta = public_ringdown_fd_converged(
        65., .7, 100., sample_rate=32768, max_sample_rate=65536,
        delta_f=1., f_max=200., comparison_band=(50., 200.))
    assert f[-1] == 200. and hp.shape == hc.shape == f.shape
    assert meta["relative_l2"] <= 1e-7


def test_convergence_rejects_nonfinite_or_unordered_controls():
    for kwargs in (
        {"sample_rate": float("inf")},
        {"max_sample_rate": float("nan")},
        {"comparison_band": (100., 100.)},
    ):
        with pytest.raises(ValueError):
            public_ringdown_fd_converged(65., .7, 100., **kwargs)
    with pytest.raises(ValueError):
        public_ringdown_fd_converged(65., .7, 100., sample_rate=256,
                                     delta_f=1., comparison_band=(50., 200.))
    with pytest.raises(ValueError):
        public_ringdown_fd_converged(65., .7, 100., sample_rate=1024,
                                     delta_f=1., f_max=100., comparison_band=(50., 101.))
