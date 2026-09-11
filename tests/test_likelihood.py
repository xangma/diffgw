"""Unit tests for differentiable likelihood and analytical detector PSD suite."""

import math
import pytest
import torch

from torchwave.detectors.psd import (
    aLIGOZeroDetHighPower,
    aLIGOEarlyHighPower,
    AdvancedVirgo,
    KAGRA,
    EinsteinTelescope,
    CosmicExplorer,
    get_psd,
)
from torchwave.inference.likelihood import (
    noise_weighted_inner_product,
    log_bessel_i0,
    WhittleLikelihood,
)
from torchwave.approximants.imrphenomd import IMRPhenomD


def test_analytical_psds():
    """Verify analytical PSD curves, positivity, and 2G vs 3G sensitivity hierarchy."""
    f = torch.linspace(20.0, 1000.0, 200, dtype=torch.float64)

    psd_aligo = aLIGOZeroDetHighPower(f)
    psd_early = aLIGOEarlyHighPower(f)
    psd_adv = AdvancedVirgo(f)
    psd_kagra = KAGRA(f)
    psd_et = EinsteinTelescope(f)
    psd_ce = CosmicExplorer(f)

    for name, psd in [
        ("aLIGO", psd_aligo),
        ("aLIGOEarly", psd_early),
        ("AdV", psd_adv),
        ("KAGRA", psd_kagra),
        ("ET", psd_et),
        ("CE", psd_ce),
    ]:
        assert torch.all(psd > 0.0), f"PSD {name} contains non-positive values"
        assert not torch.any(torch.isnan(psd)), f"PSD {name} contains NaN"

    # 3G detectors (ET, CE) must be significantly more sensitive (lower Sn) than aLIGO design
    mid_idx = 100  # ~500 Hz
    assert psd_et[mid_idx].item() < 0.1 * psd_aligo[mid_idx].item()
    assert psd_ce[mid_idx].item() < 0.05 * psd_aligo[mid_idx].item()

    # get_psd lookup
    psd_lookup = get_psd("aLIGOZeroDetHighPower", f)
    assert torch.allclose(psd_lookup, psd_aligo, atol=1e-30)


def test_noise_weighted_inner_product_properties():
    """Verify linearity, symmetry, and positive definiteness of <a | b>."""
    f = torch.linspace(20.0, 500.0, 100, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    torch.manual_seed(42)
    a = torch.randn(len(f), dtype=torch.float64) + 1j * torch.randn(len(f), dtype=torch.float64)
    b = torch.randn(len(f), dtype=torch.float64) + 1j * torch.randn(len(f), dtype=torch.float64)
    c = torch.randn(len(f), dtype=torch.float64) + 1j * torch.randn(len(f), dtype=torch.float64)

    # 1. Symmetry: <a | b> == <b | a>
    ip_ab = noise_weighted_inner_product(a, b, psd, df=df)
    ip_ba = noise_weighted_inner_product(b, a, psd, df=df)
    assert math.isclose(ip_ab.item(), ip_ba.item(), rel_tol=1e-12)

    # 2. Linearity: <a + b | c> == <a | c> + <b | c>
    ip_apb_c = noise_weighted_inner_product(a + b, c, psd, df=df)
    ip_ac = noise_weighted_inner_product(a, c, psd, df=df)
    ip_bc = noise_weighted_inner_product(b, c, psd, df=df)
    assert math.isclose(ip_apb_c.item(), (ip_ac + ip_bc).item(), rel_tol=1e-12)

    # 3. Positive definiteness: <a | a> > 0
    ip_aa = noise_weighted_inner_product(a, a, psd, df=df)
    assert ip_aa.item() > 0.0


def test_log_bessel_i0_stability():
    """Verify log_bessel_i0 evaluation and asymptotic stability for large x."""
    # Zero input
    assert math.isclose(log_bessel_i0(torch.tensor(0.0, dtype=torch.float64)).item(), 0.0, abs_tol=1e-12)

    # Large values where standard torch.i0 would overflow to +Inf
    x_large = torch.tensor([500.0, 1000.0, 5000.0, 20000.0], dtype=torch.float64)
    log_i0_vals = log_bessel_i0(x_large)

    assert not torch.any(torch.isinf(log_i0_vals))
    assert not torch.any(torch.isnan(log_i0_vals))

    # Check against asymptotic expansion: log(I0(x)) ~ x - 0.5 * log(2 * pi * x)
    for i, x_val in enumerate(x_large):
        x = x_val.item()
        asymptotic = x - 0.5 * math.log(2.0 * math.pi * x)
        assert math.isclose(log_i0_vals[i].item(), asymptotic, rel_tol=1e-4)


def test_whittle_likelihood_peak():
    """Verify that Whittle likelihood reaches maximum (ln L = 0 in noiseless case) at true parameters."""
    f = torch.linspace(20.0, 400.0, 150, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    model = IMRPhenomD()
    hp_true, _ = model(f=f, mass1=35.0, mass2=25.0, distance=100.0)

    data = {"H1": hp_true, "L1": hp_true}
    psds = {"H1": psd, "L1": psd}
    likelihood = WhittleLikelihood(data=data, psds=psds, df=df)

    # Exact injection
    ln_l_true = likelihood.log_likelihood({"H1": hp_true, "L1": hp_true})
    assert math.isclose(ln_l_true.item(), 0.0, abs_tol=1e-12)

    # Perturbed template
    hp_pert, _ = model(f=f, mass1=36.0, mass2=25.0, distance=100.0)
    ln_l_pert = likelihood.log_likelihood({"H1": hp_pert, "L1": hp_pert})
    assert ln_l_pert.item() < -100.0


def test_phase_marginalization_invariance():
    """Verify that phase-marginalized log-likelihood is invariant to template phase shifts."""
    f = torch.linspace(25.0, 300.0, 100, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    model = IMRPhenomD()
    hp_inj, _ = model(f=f, mass1=30.0, mass2=20.0, distance=100.0)

    data = {"H1": hp_inj}
    psds = {"H1": psd}
    likelihood = WhittleLikelihood(data=data, psds=psds, df=df)

    # Template with different constant phase rotations exp(i * delta_phi)
    delta_phis = [0.0, 0.5, 1.2, 2.5, 3.14]
    ln_l_vals = []

    for dphi in delta_phis:
        phase_rot = torch.complex(torch.cos(torch.tensor(dphi)), torch.sin(torch.tensor(dphi)))
        h_rotated = hp_inj * phase_rot
        ln_l = likelihood.phase_marginalized_log_likelihood({"H1": h_rotated})
        ln_l_vals.append(ln_l.item())

    # All phase-rotated evaluations should be identical
    for val in ln_l_vals[1:]:
        assert math.isclose(val, ln_l_vals[0], rel_tol=1e-6)


def test_distance_marginalization():
    """Verify distance-marginalized log-likelihood with Gauss-Legendre quadrature."""
    f = torch.linspace(25.0, 300.0, 100, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    model = IMRPhenomD()
    hp_inj, _ = model(f=f, mass1=30.0, mass2=20.0, distance=200.0)

    data = {"H1": hp_inj}
    psds = {"H1": psd}
    likelihood = WhittleLikelihood(data=data, psds=psds, df=df)

    # Template generated at reference distance 100 Mpc
    hp_ref, _ = model(f=f, mass1=30.0, mass2=20.0, distance=100.0)
    ln_l_dist = likelihood.distance_marginalized_log_likelihood(
        {"H1": hp_ref},
        d_min=50.0,
        d_max=1000.0,
        d_ref=100.0,
    )

    assert not torch.isnan(ln_l_dist)
    assert not torch.isinf(ln_l_dist)


def test_likelihood_autograd():
    """Verify finite non-zero autograd gradients of log-likelihood with respect to binary parameters."""
    f = torch.linspace(30.0, 250.0, 64, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    model = IMRPhenomD()
    hp_inj, _ = model(f=f, mass1=30.0, mass2=20.0, distance=100.0)

    data = {"H1": hp_inj}
    psds = {"H1": psd}
    likelihood = WhittleLikelihood(data=data, psds=psds, df=df)

    # Evaluate at slightly off-peak parameter to obtain non-zero gradient
    m1 = torch.tensor(31.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(20.5, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

    hp_eval, _ = model(f=f, mass1=m1, mass2=m2, chi1=chi1, distance=100.0)
    ln_l = likelihood.log_likelihood({"H1": hp_eval})
    ln_l.backward()

    for name, param in [("m1", m1), ("m2", m2), ("chi1", chi1)]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad), f"Gradient for {name} is NaN"
        assert param.grad.abs().item() > 0.0, f"Gradient for {name} is zero"
