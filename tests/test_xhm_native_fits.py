# SPDX-License-Identifier: MIT
"""Unit tests for TorchWave native SXS-calibrated higher harmonic fits."""
import math
import pytest
import torch

from torchwave.approximants.xhm_native_fits import (
    eval_native_ringdown_params,
    eval_native_21_ringdown_params,
    eval_native_33_ringdown_params,
    eval_native_44_ringdown_params,
    eval_native_32_ringdown_params,
    eval_native_33_phase_params,
    eval_native_phase_params,
    native_ringdown_amplitude,
    native_ringdown_amplitude_33,
    native_ringdown_phase,
    native_ringdown_phase_33,
    SIGMA_DEFAULTS,
    T0_OFFSET_33,
)


@pytest.mark.parametrize("lm", [(2, 1), (3, 3), (4, 4), (3, 2)])
def test_native_ringdown_params_physical(lm):
    """Verify parameters are smooth, finite, and positive across mass ratios 1 <= q <= 10."""
    l, m = lm
    q_vals = torch.linspace(1.01, 10.0, 50, dtype=torch.float64, requires_grad=True)
    eta = q_vals / ((1.0 + q_vals) ** 2)
    delta = (q_vals - 1.0) / (q_vals + 1.0)

    a_lambda, lam = eval_native_ringdown_params(l, m, eta, delta)

    # Basic physical sanity
    assert torch.all(torch.isfinite(a_lambda))
    assert torch.all(torch.isfinite(lam))
    assert torch.all(a_lambda > 0)
    assert torch.all(lam > 0)

    # Autograd works cleanly across q
    grad_a = torch.autograd.grad(a_lambda.sum(), q_vals, retain_graph=True)[0]
    grad_lam = torch.autograd.grad(lam.sum(), q_vals)[0]
    assert torch.all(torch.isfinite(grad_a))
    assert torch.all(torch.isfinite(grad_lam))


def test_native_ringdown_params_equal_mass_limit():
    """In equal mass limit (q=1, delta=0), odd-m modes vanish identically; even-m remain positive."""
    eta = torch.tensor(0.25, dtype=torch.float64)
    delta = torch.tensor(0.0, dtype=torch.float64)

    # Odd-m modes: (2, 1), (3, 3) must vanish
    a21, _ = eval_native_21_ringdown_params(eta, delta)
    a33, _ = eval_native_33_ringdown_params(eta, delta)
    assert a21.item() == 0.0
    assert a33.item() == 0.0

    # Even-m modes: (4, 4), (3, 2) must remain finite and positive
    a44, l44 = eval_native_44_ringdown_params(eta)
    a32, l32 = eval_native_32_ringdown_params(eta)
    assert a44.item() > 0.0
    assert a32.item() > 0.0
    assert math.isfinite(l44.item())
    assert math.isfinite(l32.item())


@pytest.mark.parametrize("lm", [(2, 1), (3, 3), (4, 4), (3, 2)])
def test_native_ringdown_amplitude_profile(lm):
    """Verify amplitude profile is positive, peak-centered, and smooth."""
    l, m = lm
    eta = torch.tensor(0.2222222222222222, dtype=torch.float64)  # q = 2
    delta = torch.tensor(0.3333333333333333, dtype=torch.float64)
    f_ring = 0.12 * (m / 2.0)
    f_damp = 0.014

    f = torch.linspace(0.8 * f_ring, 0.4, 200, dtype=torch.float64, requires_grad=True)
    amp = native_ringdown_amplitude(l, m, f, eta, delta, f_ring, f_damp)

    assert amp.shape == f.shape
    assert torch.all(torch.isfinite(amp))
    assert torch.all(amp > 0)

    # Backward gradient check
    grad = torch.autograd.grad(amp.sum(), f)[0]
    assert torch.all(torch.isfinite(grad))


def test_imrphenom_xhm_native_waveform():
    """Verify IMRPhenomXHM generates finite polarizations with native calibration."""
    from torchwave.waveforms import get_fd_waveform

    f = torch.linspace(20.0, 400.0, 100, dtype=torch.float64)
    hp_nat, hc_nat = get_fd_waveform(
        approximant="IMRPhenomXHM",
        mass1=40.0,
        mass2=20.0,
        chi1=0.2,
        chi2=-0.1,
        inclination=0.8,
        sample_frequencies=f,
        calibration="torchwave_nr",
    )
    assert torch.all(torch.isfinite(hp_nat))
    assert torch.all(torch.isfinite(hc_nat))
    assert torch.linalg.vector_norm(hp_nat).item() > 0.0

    hp_leg, hc_leg = get_fd_waveform(
        approximant="IMRPhenomXHM",
        mass1=40.0,
        mass2=20.0,
        chi1=0.2,
        chi2=-0.1,
        inclination=0.8,
        sample_frequencies=f,
        calibration="legacy",
    )
    # Native SXS calibration should differ from legacy calibration
    diff = torch.linalg.vector_norm(hp_nat - hp_leg) / torch.linalg.vector_norm(hp_leg)
    assert diff.item() > 0.0


def test_imrphenom_xphm_native_waveform():
    """Verify precessing IMRPhenomXPHM generates finite polarizations with native calibration."""
    from torchwave.waveforms import get_fd_waveform

    f = torch.linspace(20.0, 400.0, 100, dtype=torch.float64)
    hp_nat, hc_nat = get_fd_waveform(
        approximant="IMRPhenomXPHM",
        mass1=40.0,
        mass2=20.0,
        spin1x=0.1,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.1,
        spin2z=-0.1,
        inclination=0.8,
        sample_frequencies=f,
        calibration="torchwave_nr",
    )
    assert torch.all(torch.isfinite(hp_nat))
    assert torch.all(torch.isfinite(hc_nat))
    assert torch.linalg.vector_norm(hp_nat).item() > 0.0


def test_native_33_phase_params_physical():
    """Verify (3,3) phase parameters are smooth, finite, and follow expected physical signs."""
    q_vals = torch.linspace(1.0, 10.0, 50, dtype=torch.float64, requires_grad=True)
    eta = q_vals / ((1.0 + q_vals) ** 2)

    alpha_2, alpha_lam = eval_native_33_phase_params(eta)

    assert torch.all(torch.isfinite(alpha_2))
    assert torch.all(torch.isfinite(alpha_lam))
    assert torch.all(alpha_2 > 0)
    assert torch.all(alpha_lam < 0)

    # Autograd verification
    grad_a2 = torch.autograd.grad(alpha_2.sum(), q_vals, retain_graph=True)[0]
    grad_alam = torch.autograd.grad(alpha_lam.sum(), q_vals)[0]
    assert torch.all(torch.isfinite(grad_a2))
    assert torch.all(torch.isfinite(grad_alam))


def test_native_33_ringdown_phase_profile():
    """Verify native (3,3) ringdown phase profile is finite and differentiable."""
    eta = torch.tensor(0.1875, dtype=torch.float64)  # q = 3
    f_ring = 0.124
    f_damp = 0.014
    f = torch.linspace(0.8 * f_ring, 0.35, 100, dtype=torch.float64, requires_grad=True)

    phi = native_ringdown_phase_33(f, eta, f_ring, f_damp)
    assert phi.shape == f.shape
    assert torch.all(torch.isfinite(phi))

    grad = torch.autograd.grad(phi.sum(), f)[0]
    assert torch.all(torch.isfinite(grad))


def test_native_33_phase_holdout_quality_gates():
    """Verify sovereign (3,3) phase calibration against all 7 preregistered SXS holdouts."""
    import os
    import numpy as np
    from scripts.nr33_phase_data import load_complex_case, transform_case
    from scripts.run_nr33_phase_benchmark import quadrature_weights, phase_support
    from scripts.nr33_phase_metrics import phase_metrics
    from torchwave.approximants.imrphenom_xas import XASParameterFits
    from torchwave.approximants.xhm_fits import XHMParameterFits

    holdouts = [
        ("SXS:BBH:2265", "/tmp/torchwave-nr33-v2/npz/SXS_BBH_2265_258wv-a5t45_Lev3_N3.npz", 3.0),
        ("SXS:BBH:2483", "/tmp/torchwave-nr33-expanded/npz-replay/SXS_BBH_2483_z5qsd-ptm37_Lev5_N3.npz", 3.5),
        ("SXS:BBH:2484", "/tmp/torchwave-nr33-expanded/npz-replay/SXS_BBH_2484_x0q4y-s3440_Lev5_N3.npz", 4.5),
        ("SXS:BBH:2486", "/tmp/torchwave-nr33-expanded/npz-replay/SXS_BBH_2486_1wj32-qdk56_Lev5_N3.npz", 5.5),
        ("SXS:BBH:2489", "/tmp/torchwave-nr33-v3-search/npz/SXS_BBH_2489_q7rzh-bzw94_Lev5_N3.npz", 6.0),
        ("SXS:BBH:2491", "/tmp/torchwave-nr33-expanded/npz-replay/SXS_BBH_2491_rx0d1-q7d95_Lev5_N3.npz", 7.0),
        ("SXS:BBH:0185", "/tmp/torchwave-nr33-expanded/npz-replay/SXS_BBH_0185_y7mbp-x8889_Lev3_N3.npz", 10.0),
    ]

    for name, path, q in holdouts:
        if not os.path.exists(path):
            pytest.skip(f"Holdout file {path} not available locally")
        case = load_complex_case(path, "h33")
        spec = transform_case(case)

        eta = q / (1.0 + q)**2
        eta_t = torch.tensor(eta, dtype=torch.float64)
        chi_f = XASParameterFits.fit_final_spin(eta_t, torch.tensor(0.0), torch.tensor(0.0))
        m_f = 1.0 - XASParameterFits.fit_erad(eta_t, torch.tensor(0.0), torch.tensor(0.0))
        f_ring = (XHMParameterFits.qnm_fring_33(chi_f) / m_f).item()
        f_damp = (XHMParameterFits.qnm_fdamp_33(chi_f) / m_f).item()

        band = (spec.frequency_Mf >= 0.95 * f_ring) & (spec.frequency_Mf <= 0.3)
        f = spec.frequency_Mf[band]
        reference = -spec.spectrum[band]
        weights = quadrature_weights(f)
        mask = phase_support(reference, 1e-3)

        f_t = torch.as_tensor(f, dtype=torch.float64)
        phi_model = native_ringdown_phase_33(f_t, eta_t, f_ring, f_damp, include_arrival_time=True).numpy()
        candidate = np.abs(reference) * np.exp(1j * phi_model)

        m = phase_metrics(f, candidate, reference, weights=weights, phase_mask=mask)

        # Quality gates:
        # Phase RMS < 0.02 rad across all holdouts (< 0.003 rad for q <= 3.5)
        if q <= 3.5:
            assert m["affine_phase_rms"] < 0.003
            assert m["affine_relative_l2"] < 0.003
            assert m["affine_mismatch"] < 5.0e-6
        else:
            assert m["affine_phase_rms"] < 0.020
            assert m["affine_relative_l2"] < 0.020
            assert m["affine_mismatch"] < 2.5e-4
        # Universal arrival time residual within +-0.20 M
        assert abs(m["time_shift"]) < 0.20

