"""Independent behavioral checks for the native IMRPhenomT time model.

The LALSuite checks in this file are deliberately limited to conventions and
symmetries.  They are not coefficient or calibration fits.
"""

import numpy as np
import pytest
import torch

from torchwave.approximants.imrphenomt import IMRPhenomT


def _times(n=4301, dt=1.0 / 4096.0, start=-1.0):
    return start + dt * torch.arange(n, dtype=torch.float64)


def _wave(model, times, **kwargs):
    return model.time_domain(times=times, mass1=30.0, mass2=20.0, distance=400.0, **kwargs)


def test_native_td_call_alias_and_real_output():
    model = IMRPhenomT()
    times = _times()
    hp, hc = _wave(model, times, chi1=0.4, chi2=-0.2)
    hp_call, hc_call = model(times, mass1=30.0, mass2=20.0, chi1=0.4, chi2=-0.2, distance=400.0)

    assert hp.shape == hc.shape == times.shape
    assert hp.dtype == hc.dtype == torch.float64
    assert torch.isfinite(hp).all() and torch.isfinite(hc).all()
    assert torch.any(torch.abs(hp) > 0)
    torch.testing.assert_close(hp_call, hp)
    torch.testing.assert_close(hc_call, hc)


def test_native_td_broadcasting_and_mass_spin_exchange():
    model = IMRPhenomT()
    times = _times(n=2048)
    masses1 = torch.tensor([30.0, 35.0], dtype=torch.float64)
    masses2 = torch.tensor([20.0, 18.0], dtype=torch.float64)
    chi1 = torch.tensor([0.4, -0.2], dtype=torch.float64)
    chi2 = torch.tensor([-0.2, 0.1], dtype=torch.float64)

    hp_b, hc_b = model.time_domain(times, masses1, masses2, chi1=chi1, chi2=chi2, distance=400.0)
    assert hp_b.shape == hc_b.shape == (2, times.numel())
    for i in range(2):
        hp_s, hc_s = model.time_domain(times, masses1[i], masses2[i], chi1=chi1[i], chi2=chi2[i], distance=400.0)
        torch.testing.assert_close(hp_b[i], hp_s, rtol=2e-12, atol=1e-30)
        torch.testing.assert_close(hc_b[i], hc_s, rtol=2e-12, atol=1e-30)

    hp_a, hc_a = model.time_domain(times, 30.0, 20.0, chi1=0.4, chi2=-0.2, distance=400.0)
    hp_s, hc_s = model.time_domain(times, 20.0, 30.0, chi1=-0.2, chi2=0.4, distance=400.0)
    torch.testing.assert_close(hp_a, hp_s, rtol=2e-11, atol=1e-30)
    torch.testing.assert_close(hc_a, hc_s, rtol=2e-11, atol=1e-30)


def test_native_td_autograd():
    model = IMRPhenomT()
    times = _times(n=2048)
    m1 = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
    hp, hc = model.time_domain(times, m1, m2, chi1=chi1, chi2=chi2, distance=400.0)
    (hp.square().sum() + hc.square().sum()).backward()
    for value in (m1, m2, chi1, chi2):
        assert value.grad is not None
        assert torch.isfinite(value.grad)
        assert value.grad != 0


def test_fref_zero_is_fmin_and_reference_phase_changes_only_phase():
    model = IMRPhenomT()
    times = _times(n=2048)
    hp0, hc0 = _wave(model, times, chi1=0.4, chi2=-0.2, f_ref=0.0)
    hpm, hcm = _wave(model, times, chi1=0.4, chi2=-0.2, f_ref=20.0)
    torch.testing.assert_close(hp0, hpm, rtol=2e-11, atol=1e-30)
    torch.testing.assert_close(hc0, hcm, rtol=2e-11, atol=1e-30)

    hp50, hc50 = _wave(model, times, chi1=0.4, chi2=-0.2, f_ref=50.0)
    # A reference-frequency change is a common complex phase rotation.  Check
    # the invariant |h_plus - i h_cross| instead of individual projections.
    h0, hm, h50 = hp0 - 1j * hc0, hpm - 1j * hcm, hp50 - 1j * hc50
    torch.testing.assert_close(torch.abs(h0), torch.abs(hm), rtol=2e-11, atol=1e-30)
    torch.testing.assert_close(torch.abs(h0), torch.abs(h50), rtol=2e-11, atol=1e-30)


def test_lalsuite_td_time_and_face_on_conventions():
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    m = lal.MSUN_SI
    distance = 400.0e6 * lal.PC_SI
    dt = 1.0 / 4096.0
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        30 * m,
        20 * m,
        0.0,
        0.0,
        0.4,
        0.0,
        0.0,
        -0.2,
        distance,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        dt,
        20.0,
        0.0,
        lal.CreateDict(),
        lalsim.IMRPhenomT,
    )
    plus = np.asarray(hp.data.data)
    cross = np.asarray(hc.data.data)
    peak_time = float(hp.epoch) + dt * int(np.argmax(np.abs(plus)))
    assert abs(peak_time) < 2.0 * dt

    # Positive-frequency face-on polarizations are in the local -i convention.
    plus_f = np.fft.rfft(plus)
    cross_f = np.fft.rfft(cross)
    keep = np.abs(plus_f) > 0.2 * np.max(np.abs(plus_f))
    ratio = cross_f[keep] / plus_f[keep]
    assert abs(np.median(ratio.real)) < 0.03
    assert np.median(ratio.imag) < -0.95


def test_native_td_matches_lalsuite_reference_waveform():
    """Check numerical fidelity against the independent LAL runtime oracle.

    This compares the published calibrated waveform output directly; it does
    not derive or fit any coefficients from the oracle.
    """
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    m = lal.MSUN_SI
    distance_mpc = 400.0
    dt = 1.0 / 4096.0
    hp_lal, hc_lal = lalsim.SimInspiralChooseTDWaveform(
        30 * m,
        20 * m,
        0.0,
        0.0,
        0.4,
        0.0,
        0.0,
        -0.2,
        distance_mpc * 1.0e6 * lal.PC_SI,
        0.7,
        0.2,
        0.0,
        0.0,
        0.0,
        dt,
        20.0,
        20.0,
        lal.CreateDict(),
        lalsim.IMRPhenomT,
    )
    n = len(hp_lal.data.data)
    epoch = float(hp_lal.epoch)
    times = torch.arange(n, dtype=torch.float64) * dt + epoch
    hp, hc = IMRPhenomT().time_domain(
        times, 30.0, 20.0, chi1=0.4, chi2=-0.2, distance=distance_mpc, inclination=0.7, phic=0.2, f_ref=20.0
    )
    hp_ref = torch.as_tensor(np.asarray(hp_lal.data.data), dtype=torch.float64)
    hc_ref = torch.as_tensor(np.asarray(hc_lal.data.data), dtype=torch.float64)
    denom = max(float(torch.linalg.vector_norm(hp_ref)), float(torch.linalg.vector_norm(hc_ref)))
    rel = max(float(torch.linalg.vector_norm(hp - hp_ref)), float(torch.linalg.vector_norm(hc - hc_ref))) / denom
    assert rel < 1.0e-5


def test_lalsuite_mass_exchange_and_fref_zero_semantics():
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    m = lal.MSUN_SI
    distance = 400.0e6 * lal.PC_SI
    dt = 1.0 / 4096.0

    def generate(m1, m2, chi1, chi2, f_ref):
        return lalsim.SimInspiralChooseTDWaveform(
            m1 * m,
            m2 * m,
            0.0,
            0.0,
            chi1,
            0.0,
            0.0,
            chi2,
            distance,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            dt,
            20.0,
            f_ref,
            lal.CreateDict(),
            lalsim.IMRPhenomT,
        )

    base = generate(30.0, 20.0, 0.4, -0.2, 0.0)
    exchanged = generate(20.0, 30.0, -0.2, 0.4, 20.0)
    for a, b in zip(base, exchanged):
        np.testing.assert_array_equal(np.asarray(a.data.data), np.asarray(b.data.data))


def test_native_amplitude_join_value_and_slope_continuity():
    """The calibrated amplitude ansatz is C1 at inspiral/merger joins."""
    from torchwave.approximants._phenomt_amplitude import _merger_basis, _pn

    model = IMRPhenomT()
    dtype = torch.float64
    state = model._coeff(
        torch.tensor(30.0, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(0.4, dtype=dtype),
        torch.tensor(-0.2, dtype=dtype),
    )
    pn, pseudo, merger, alpha, c1, c2, c3, c4 = state[8]

    def omega(u):
        return model._omega_insp(u, state[1], state[2], state[5], state[6], state[7])

    def inspiral(u):
        x = (omega(u) / 2) ** (2 / 3)
        return torch.abs(_pn(x, pn, pseudo))

    def middle(u):
        return torch.sum(_merger_basis(u, alpha) * merger, dim=-1)

    def ringdown(u):
        return torch.exp(-alpha * u) * (c1 * torch.tanh(c2 * u + c3) + c4)

    torch.testing.assert_close(
        inspiral(torch.tensor([-150.0], dtype=dtype)),
        middle(torch.tensor([-150.0], dtype=dtype)),
        rtol=2e-8,
        atol=2e-10,
    )
    torch.testing.assert_close(
        middle(torch.tensor([0.0], dtype=dtype)), ringdown(torch.tensor([0.0], dtype=dtype)), rtol=2e-8, atol=2e-10
    )
    # Differentiate each active branch at the exact join, avoiding curvature
    # contamination from one-sided finite differences.
    for boundary, left_branch, right_branch in ((-150.0, inspiral, middle), (0.0, middle, ringdown)):
        u = torch.tensor([boundary], dtype=dtype, requires_grad=True)
        dleft = torch.autograd.grad(left_branch(u).sum(), u, create_graph=False)[0]
        dright = torch.autograd.grad(right_branch(u).sum(), u, create_graph=False)[0]
        torch.testing.assert_close(dleft, dright, rtol=2.0e-6, atol=2.0e-9)


def test_native_phase_derivative_matches_frequency_each_ansatz():
    """The phase primitive differentiates to the dimensionless angular frequency."""
    model = IMRPhenomT()
    dtype = torch.float64
    state = model._coeff(
        torch.tensor(30.0, dtype=dtype),
        torch.tensor(20.0, dtype=dtype),
        torch.tensor(0.4, dtype=dtype),
        torch.tensor(-0.2, dtype=dtype),
    )
    h = 1.0e-4
    # Stay away from the piecewise joins at t=-26.98/eta and t=0.
    for u0 in (-1000.0, -200.0, -1.0, 0.1, 10.0):
        u = torch.tensor([u0 - h, u0 + h], dtype=dtype)
        omega, phase, _ = model._evaluate(u, state)
        dphase = (phase[1] - phase[0]) / (2 * h)
        torch.testing.assert_close(dphase, omega.mean(), rtol=2.0e-6, atol=2.0e-8)


def test_native_phase_sensitive_intrinsic_gradients_match_finite_difference():
    """Intrinsic gradients remain correct when the objective sees phase."""
    model = IMRPhenomT()
    times = _times(n=257, dt=1.0 / 2048.0, start=-0.08)
    base = torch.tensor([30.0, 20.0, 0.4, -0.2], dtype=torch.float64)
    weights = torch.linspace(-1.0, 1.0, times.numel(), dtype=torch.float64)

    def objective(values):
        hp, hc = model.time_domain(
            times, values[0], values[1], chi1=values[2], chi2=values[3], distance=400.0, phic=0.23
        )
        return ((hp * (1.0 + weights) + hc * (1.0 - 0.3 * weights)) * 1.0e40).sum()

    values = base.clone().requires_grad_(True)
    objective(values).backward()
    analytic = values.grad.detach()
    numeric = []
    eps = 1.0e-5
    for idx in range(4):
        vp, vm = base.clone(), base.clone()
        vp[idx] += eps
        vm[idx] -= eps
        numeric.append(((objective(vp) - objective(vm)) / (2 * eps)).detach())
    numeric = torch.stack(numeric)
    torch.testing.assert_close(analytic, numeric, rtol=3e-5, atol=2e-3)


def test_time_frequency_roundtrip_and_long_inspiral_gradient():
    """The physical root is valid for merger and long low-frequency inspiral."""
    model = IMRPhenomT()
    for mass1, mass2, frequency, positive_ringdown in (
        (30.0, 20.0, 80.0, False),
        (10.0, 1.0, 40.0, False),
        (1.4, 1.4, 10.0, False),
        (30.0, 25.0, 300.0, True),
    ):
        t = model.time_at_frequency(frequency, mass1, mass2)
        recovered = model.frequency(t, mass1, mass2)
        torch.testing.assert_close(
            recovered.squeeze(), torch.as_tensor(frequency, dtype=torch.float64), rtol=2e-7, atol=2e-8
        )
        assert torch.isfinite(t)
        assert bool(t > 0) is positive_ringdown
    mass = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    t = model.time_at_frequency(10.0, mass, mass)
    (gradient,) = torch.autograd.grad(t, mass)
    assert torch.isfinite(gradient) and gradient != 0
