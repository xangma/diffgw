"""Tests for clean-room IMRPhenomXAS aligned-spin 4th-generation approximant.

Literature citations:
- Pratten et al. (2020), Phys. Rev. D 102, 064001 (arXiv:2001.11412)
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914)
- Berti et al. (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
"""

import math
import pytest
import numpy as np
import torch

from torchwave.waveforms import get_fd_waveform, _APPROXIMANTS
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
from torchwave.approximants.imrphenom_hm import IMRPhenomXHM

try:
    import lal
    import lalsimulation as lalsim
    from pycbc.filter.matchedfilter import optimized_match
    from pycbc.types import FrequencySeries
    HAS_LAL = True
except ImportError:
    HAS_LAL = False


def test_imrphenom_xas_registration():
    """Verify IMRPhenomXAS is registered under canonical names."""
    for name in ["imrphenomxas", "phenomxas", "xas"]:
        assert name in _APPROXIMANTS
        hp, hc = get_fd_waveform(
            approximant=name,
            mass1=30.0,
            mass2=20.0,
            sample_frequencies=torch.linspace(30.0, 300.0, 50, dtype=torch.float64),
        )
        assert hp is not None and hc is not None


@pytest.mark.skipif(not HAS_LAL, reason="LALSimulation / PyCBC required for parity verification")
@pytest.mark.parametrize(
    "m1, m2, chi1, chi2, min_match",
    [
        (35.0, 30.0, 0.2, -0.3, 0.99),
        (50.0, 15.0, 0.6, 0.4, 0.99),
        (60.0, 7.5, -0.4, 0.1, 0.99),
        (75.0, 5.0, 0.5, -0.2, 0.99),
    ],
)
def test_imrphenom_xas_lal_smoke_overlap(m1, m2, chi1, chi2, min_match):
    """Verify diagnostic overlap (smoke test, match >= min_match) against LALSimulation IMRPhenomXAS across parameter space.
    
    Note: Strict parity gates (1e-6 complex relative L2, 1e-8 mismatch) are evaluated in parity_harness.py.
    """
    dist_mpc = 500.0
    f_min = 20.0
    f_max = 512.0
    delta_f = 0.125

    lal_approx = lalsim.GetApproximantFromString("IMRPhenomXAS")
    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        0.0, 0.0, chi1,
        0.0, 0.0, chi2,
        dist_mpc * 1e6 * lal.PC_SI,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        delta_f,
        f_min,
        f_max,
        f_min,
        None,
        lal_approx,
    )

    n_pts = int(f_max / delta_f) + 1
    f_grid = torch.arange(0, n_pts, dtype=torch.float64) * delta_f
    f_mask = (f_grid >= f_min) & (f_grid <= f_max)
    f_eval = f_grid[f_mask]

    hp_tw, _ = get_fd_waveform(
        approximant="IMRPhenomXAS",
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        sample_frequencies=f_eval,
        distance=dist_mpc,
        f_ref=f_min,
    )

    s_lal = FrequencySeries(hp_lal.data.data, delta_f=delta_f)
    s_tw = FrequencySeries(np.zeros(len(s_lal), dtype=np.complex128), delta_f=delta_f)
    idx_start = int(f_min / delta_f)
    s_tw.data[idx_start:idx_start + len(hp_tw)] = hp_tw.detach().cpu().numpy()

    overlap, _ = optimized_match(s_lal, s_tw, low_frequency_cutoff=f_min, high_frequency_cutoff=f_max)
    assert overlap >= min_match, f"IMRPhenomXAS match = {overlap:.6f} is below {min_match}"


def test_imrphenom_xas_batched_evaluation():
    """Verify batched evaluation consistency across batch dimensions with physical strain scaling."""
    f = torch.linspace(20.0, 400.0, 200, dtype=torch.float64)
    model = IMRPhenomXAS()

    m1_b = torch.tensor([30.0, 45.0], dtype=torch.float64)
    m2_b = torch.tensor([25.0, 15.0], dtype=torch.float64)
    chi1_b = torch.tensor([0.2, -0.4], dtype=torch.float64)
    chi2_b = torch.tensor([-0.1, 0.3], dtype=torch.float64)

    hp_b, hc_b = model(
        sample_frequencies=f,
        mass1=m1_b,
        mass2=m2_b,
        chi1=chi1_b,
        chi2=chi2_b,
    )

    assert hp_b.shape == (2, 200)
    assert hc_b.shape == (2, 200)
    assert torch.all(torch.isfinite(hp_b))
    assert torch.all(torch.isfinite(hc_b))

    # Compare first batch element with single evaluation
    hp_0, hc_0 = model(
        sample_frequencies=f,
        mass1=30.0,
        mass2=25.0,
        chi1=0.2,
        chi2=-0.1,
    )
    scale = torch.max(torch.abs(hp_0)).item()
    assert scale > 0.0, "Waveform strain scale must be non-zero"
    assert torch.allclose(hp_b[0], hp_0, rtol=1e-12, atol=1e-12 * scale)
    assert torch.allclose(hc_b[0], hc_0, rtol=1e-12, atol=1e-12 * scale)


def test_imrphenom_xas_differentiability():
    """Verify PyTorch autograd gradients agree with converged finite differences on a phase-sensitive loss."""
    f = torch.linspace(30.0, 300.0, 50, dtype=torch.float64)
    model = IMRPhenomXAS()

    m1 = torch.tensor(36.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(29.0, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)

    hp, _ = model(sample_frequencies=f, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2)
    loss = torch.sum(hp.real)
    loss.backward()

    base_args = {"sample_frequencies": f, "mass1": 36.0, "mass2": 29.0, "chi1": 0.3, "chi2": -0.2}
    param_checks = {
        "m1": (m1, "mass1", 36.0, 1e-4),
        "m2": (m2, "mass2", 29.0, 1e-4),
        "chi1": (chi1, "chi1", 0.3, 1e-4),
        "chi2": (chi2, "chi2", -0.2, 1e-4),
    }

    for p_name, (param, kw_key, val, eps) in param_checks.items():
        assert param.grad is not None, f"Gradient for {p_name} is None"
        assert torch.isfinite(param.grad), f"Gradient for {p_name} is non-finite: {param.grad}"
        auto_grad = param.grad.item()
        assert abs(auto_grad) > 0.0, f"Gradient for {p_name} is zero"

        # Central finite difference at eps
        kw_p1 = dict(base_args)
        kw_m1 = dict(base_args)
        kw_p1[kw_key] = val + eps
        kw_m1[kw_key] = val - eps
        hp_p1, _ = model(**kw_p1)
        hp_m1, _ = model(**kw_m1)
        num_grad1 = ((torch.sum(hp_p1.real) - torch.sum(hp_m1.real)) / (2.0 * eps)).item()

        # Central finite difference at eps/2 to verify convergence
        eps2 = eps / 2.0
        kw_p2 = dict(base_args)
        kw_m2 = dict(base_args)
        kw_p2[kw_key] = val + eps2
        kw_m2[kw_key] = val - eps2
        hp_p2, _ = model(**kw_p2)
        hp_m2, _ = model(**kw_m2)
        num_grad2 = ((torch.sum(hp_p2.real) - torch.sum(hp_m2.real)) / (2.0 * eps2)).item()

        assert abs(num_grad2 - num_grad1) / abs(num_grad2) < 1e-4, f"Finite difference not converged for {p_name}"
        rel_diff = abs(auto_grad - num_grad2) / abs(num_grad2)
        assert rel_diff < 0.02, f"Autograd derivative {auto_grad} differs from numerical {num_grad2} for {p_name} (rel {rel_diff})"


def test_imrphenom_xas_in_xhm_carrier():
    """Verify IMRPhenomXHM works seamlessly with XAS carrier."""
    f = torch.linspace(20.0, 400.0, 200, dtype=torch.float64)
    xhm = IMRPhenomXHM(carrier="XAS")
    hp, hc = xhm(
        sample_frequencies=f,
        mass1=40.0,
        mass2=20.0,
        chi1=0.3,
        chi2=-0.1,
        inclination=math.pi / 3.0,
    )
    assert hp.shape == (200,)
    assert torch.all(torch.isfinite(hp))
    assert torch.all(torch.isfinite(hc))
    assert torch.max(torch.abs(hp)) > 0.0
