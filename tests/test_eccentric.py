"""Unit tests for EccentricTaylorF2 approximant."""

import math
import pytest
import torch

from torchwave import get_fd_waveform
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric


def test_eccentric_zero_limit():
    """Verify that e0 = 0 collapses to quasi-circular TaylorF2 to machine precision."""
    f = torch.linspace(20.0, 300.0, 150, dtype=torch.float64)
    m1, m2 = 1.4, 1.4
    s1, s2 = 0.05, -0.02
    dist = 50.0
    inc = 0.5
    f_ref = 20.0

    t2_circ = TaylorF2()
    t2_ecc = TaylorF2_Eccentric()

    hp_circ, hc_circ = t2_circ(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=s1,
        chi2=s2,
        distance=dist,
        inclination=inc,
        f_ref=f_ref,
    )

    hp_ecc, hc_ecc = t2_ecc(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=s1,
        chi2=s2,
        e0=0.0,
        distance=dist,
        inclination=inc,
        f_ref=f_ref,
        f0_ecc=f_ref,
    )

    diff_p = torch.max(torch.abs(hp_ecc - hp_circ)).item()
    diff_c = torch.max(torch.abs(hc_ecc - hc_circ)).item()
    max_amp = torch.max(torch.abs(hp_circ)).item()

    assert diff_p / max_amp < 1e-12, f"h_plus relative diff {diff_p / max_amp} exceeds 1e-12"
    assert diff_c / max_amp < 1e-12, f"h_cross relative diff {diff_c / max_amp} exceeds 1e-12"


def test_eccentric_phase_correction_properties():
    """Verify negative dephasing and quadratic e0^2 scaling of Moore et al. (2016) correction."""
    f = torch.linspace(30.0, 200.0, 50, dtype=torch.float64)
    m1 = torch.tensor(1.5, dtype=torch.float64)
    m2 = torch.tensor(1.3, dtype=torch.float64)
    f0 = torch.tensor(30.0, dtype=torch.float64)

    e1 = torch.tensor(0.05, dtype=torch.float64)
    e2 = torch.tensor(0.10, dtype=torch.float64)

    dpsi_1 = TaylorF2_Eccentric.eccentric_phase_correction(f, m1, m2, e1, f0)
    dpsi_2 = TaylorF2_Eccentric.eccentric_phase_correction(f, m1, m2, e2, f0)

    # Dephasing must be negative (Moore et al. 2016 Eq. 4.17 has leading minus sign)
    assert torch.all(dpsi_1 < 0.0)
    assert torch.all(dpsi_2 < 0.0)

    # Scaling must be quadratic with e0: dpsi(e2) / dpsi(e1) == (0.10 / 0.05)^2 = 4.0
    ratio = dpsi_2 / dpsi_1
    expected_ratio = (e2 / e1) ** 2
    assert torch.allclose(ratio, expected_ratio, atol=1e-12)


def test_eccentric_amplitude_correction_properties():
    """Verify amplitude suppression factor from Moore et al. (2016) Eq. 4.20."""
    f = torch.linspace(30.0, 200.0, 50, dtype=torch.float64)
    f0 = torch.tensor(30.0, dtype=torch.float64)

    # e0 = 0 should give factor 1.0 identically
    factor_0 = TaylorF2_Eccentric.eccentric_amplitude_correction(f, torch.tensor(0.0, dtype=torch.float64), f0)
    assert torch.all(factor_0 == 1.0)

    # e0 > 0 should suppress amplitude slightly (factor < 1.0)
    factor_e = TaylorF2_Eccentric.eccentric_amplitude_correction(f, torch.tensor(0.1, dtype=torch.float64), f0)
    assert torch.all(factor_e < 1.0)
    assert torch.all(factor_e > 0.9)


def test_eccentric_batching():
    """Verify batched evaluation across masses, spins, and eccentricity."""
    B = 4
    N = 80
    f = torch.linspace(25.0, 250.0, N, dtype=torch.float64)

    m1_b = torch.tensor([1.4, 1.6, 2.0, 1.5], dtype=torch.float64)
    m2_b = torch.tensor([1.2, 1.3, 1.4, 1.1], dtype=torch.float64)
    s1_b = torch.tensor([0.02, -0.01, 0.05, 0.0], dtype=torch.float64)
    s2_b = torch.tensor([-0.01, 0.03, 0.0, -0.04], dtype=torch.float64)
    e0_b = torch.tensor([0.0, 0.03, 0.08, 0.15], dtype=torch.float64)
    inc_b = torch.tensor([0.0, 0.3, 0.6, 0.9], dtype=torch.float64)

    model = TaylorF2_Eccentric()
    hp_b, hc_b = model(
        f=f,
        mass1=m1_b,
        mass2=m2_b,
        chi1=s1_b,
        chi2=s2_b,
        e0=e0_b,
        inclination=inc_b,
    )

    assert hp_b.shape == (B, N)
    assert hc_b.shape == (B, N)

    for b in range(B):
        hp_s, hc_s = model(
            f=f,
            mass1=m1_b[b],
            mass2=m2_b[b],
            chi1=s1_b[b],
            chi2=s2_b[b],
            e0=e0_b[b],
            inclination=inc_b[b],
        )
        assert torch.allclose(hp_b[b], hp_s, atol=1e-25)
        assert torch.allclose(hc_b[b], hc_s, atol=1e-25)


def test_eccentric_autograd():
    """Verify finite non-zero gradients with respect to m1, m2, chi1, chi2, e0, and inc."""
    f = torch.linspace(30.0, 150.0, 40, dtype=torch.float64)

    m1 = torch.tensor(1.45, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
    s1 = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
    s2 = torch.tensor(-0.02, dtype=torch.float64, requires_grad=True)
    e0 = torch.tensor(0.06, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.45, dtype=torch.float64, requires_grad=True)

    hp, hc = TaylorF2_Eccentric()(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=s1,
        chi2=s2,
        e0=e0,
        inclination=inc,
    )

    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()

    for name, param in [("m1", m1), ("m2", m2), ("s1", s1), ("s2", s2), ("e0", e0), ("inc", inc)]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad), f"Gradient for {name} is NaN"
        assert param.grad.abs().item() > 0.0, f"Gradient for {name} is zero"


def test_eccentric_get_fd_waveform():
    """Verify dispatch through get_fd_waveform high-level entrypoint."""
    f = torch.linspace(25.0, 200.0, 64, dtype=torch.float64)
    hp, hc = get_fd_waveform(
        "EccentricTaylorF2",
        mass1=1.4,
        mass2=1.3,
        chi1=0.02,
        e0=0.05,
        sample_frequencies=f,
    )
    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert not torch.all(hp == 0.0)


def test_ecc_default_on_dc_grid():
    """Verify eccentricity reference resolution independent of grid DC padding."""
    f = torch.arange(0.0, 500.25, 0.25, dtype=torch.float64)
    cfg = dict(mass1=1.4, mass2=1.4, e0=0.05, f_lower=30.0, sample_frequencies=f)
    a = get_fd_waveform("TaylorF2_Eccentric", **cfg)[0]
    b = get_fd_waveform("TaylorF2_Eccentric", **cfg, f0_ecc=0.25)[0]
    c = get_fd_waveform("TaylorF2_Eccentric", **cfg, f0_ecc=30.0)[0]

    # a must NOT equal b (defaulting to 0.25 Hz DC grid spacing)
    assert not torch.equal(a, b), "f0_ecc incorrectly defaulted to 0.25 Hz DC grid step"

    # In-band signal of a must match c (explicit f0_ecc=30.0 Hz) to machine precision
    sel = f >= 30.0
    rel_l2 = float(torch.linalg.norm(c[sel] - a[sel]) / torch.linalg.norm(c[sel]))
    assert rel_l2 == 0.0, f"In-band waveform differs from explicit f0_ecc=30 Hz: rel_l2 = {rel_l2}"


def test_ecc_vector_fref():
    """Verify batched reference frequency does not raise scalar conversion errors."""
    f = torch.arange(0.0, 500.25, 0.25, dtype=torch.float64)
    m1 = torch.tensor([1.4, 1.7], dtype=torch.float64)
    m2 = torch.tensor([1.3, 1.2], dtype=torch.float64)
    e0 = torch.tensor([0.05, 0.03], dtype=torch.float64)
    f_ref = torch.tensor([30.0, 50.0], dtype=torch.float64)

    hp, hc = get_fd_waveform(
        "TaylorF2_Eccentric",
        mass1=m1,
        mass2=m2,
        e0=e0,
        f_ref=f_ref,
        sample_frequencies=f,
    )
    assert hp.shape == (2, len(f))
    assert hc.shape == (2, len(f))

    # Compare with scalar evaluations
    for b in range(2):
        hp_s, hc_s = get_fd_waveform(
            "TaylorF2_Eccentric",
            mass1=m1[b].item(),
            mass2=m2[b].item(),
            e0=e0[b].item(),
            f_ref=f_ref[b].item(),
            sample_frequencies=f,
        )
        rel_l2_p = float(torch.linalg.norm(hp[b] - hp_s) / torch.linalg.norm(hp_s))
        rel_l2_c = float(torch.linalg.norm(hc[b] - hc_s) / torch.linalg.norm(hc_s))
        assert rel_l2_p < 1e-12, f"Batch {b} plus relative L2 {rel_l2_p:.3e} exceeds 1e-12"
        assert rel_l2_c < 1e-12, f"Batch {b} cross relative L2 {rel_l2_c:.3e} exceeds 1e-12"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ecc_dc_gradient(dtype):
    """Verify forward and autograd finite at DC (f=0) for float32 and float64."""
    e = torch.tensor(0.05, dtype=dtype, requires_grad=True)
    f = torch.tensor([0.0, 30.0, 50.0, 100.0], dtype=dtype)
    h = get_fd_waveform(
        "TaylorF2_Eccentric",
        mass1=1.4,
        mass2=1.4,
        e0=e,
        f0_ecc=30.0,
        f_ref=30.0,
        sample_frequencies=f,
        dtype=dtype,
    )[0]
    assert torch.isfinite(h).all(), f"Forward waveform contains non-finite values for {dtype}"
    grad, = torch.autograd.grad((h.real * 1e23).sum(), e)
    assert torch.isfinite(grad), f"Autograd gradient is non-finite for {dtype}"


def test_ecc_model_alias_reference():
    """Verify EccentricTaylorF2 alias behaves identically to TaylorF2_Eccentric with f_lower."""
    f = torch.arange(0.0, 500.25, 0.25, dtype=torch.float64)
    ecc_cfg = dict(mass1=1.4, mass2=1.4, e0=0.05, f_lower=30.0, sample_frequencies=f)
    canonical = get_fd_waveform("TaylorF2_Eccentric", **ecc_cfg)[0]
    alias = get_fd_waveform("EccentricTaylorF2", **ecc_cfg)[0]
    sel = f >= 30.0
    rel_l2 = float(torch.linalg.norm(alias[sel] - canonical[sel]) / torch.linalg.norm(canonical[sel]))
    assert rel_l2 == 0.0, f"Alias differs from canonical: rel_l2 = {rel_l2}"
    assert torch.equal(alias, canonical)

