"""Independent source-audited checks for the pass-11 XHM normalization."""

import math

import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC, PI
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
from torchwave.approximants import imrphenom_xhm as xhm_module
from torchwave.approximants.imrphenom_xhm import (
    _carrier_h22_ratio,
    _spherical_32_from_spheroidal,
)
from torchwave.approximants.xhm_fits import XHMParameterFits


def test_xas_carrier_matches_paper_ratio_after_a0_cancellation():
    """The paper's N factors cancel between physical h22 and A^0_22."""
    f = torch.tensor([100.0], dtype=torch.float64)
    mass1 = torch.tensor(35.0, dtype=torch.float64)
    mass2 = torch.tensor(20.0, dtype=torch.float64)
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    m_sec = (mass1 + mass2) * MSUN_SEC
    distance = m_sec * C_SI / MPC_SI

    strain, _, phase = IMRPhenomXAS()( 
        f=f,
        mass1=mass1,
        mass2=mass2,
        chi1=torch.tensor(0.2, dtype=torch.float64),
        chi2=torch.tensor(-0.1, dtype=torch.float64),
        distance=distance,
        inclination=0.0,
        phic=0.0,
        tc=0.0,
        return_phase=True,
    )
    native_a0 = (
        torch.sqrt(5.0 * eta / 24.0)
        / PI ** (2.0 / 3.0)
        / m_sec ** (1.0 / 6.0)
        * f ** (-7.0 / 6.0)
    )
    paper_n = math.sqrt(16.0 * math.pi / 5.0)
    # Eq. (4.5), evaluated at the same D=M convention as the carrier probe.
    paper_a0 = (
        torch.sqrt(2.0 * eta / 3.0)
        * PI ** (-1.0 / 6.0)
        * m_sec ** (-1.0 / 6.0)
        * f ** (-7.0 / 6.0)
    )

    result = _carrier_h22_ratio(strain, phase, phase, native_a0)
    expected = (paper_n * torch.abs(strain) / paper_a0).to(result.dtype)
    torch.testing.assert_close(result, expected, rtol=1e-14, atol=1e-14)


def test_xhm_32_mixing_receives_normalized_carrier(monkeypatch):
    """The production (3,2) probe passes the cancelled carrier ratio to mixing."""
    helper_values = []
    mixing_values = []
    original_helper = xhm_module._carrier_h22_ratio
    original_mixing = xhm_module._spherical_32_from_spheroidal

    def capture_helper(*args, **kwargs):
        result = original_helper(*args, **kwargs)
        helper_values.append(result.detach().clone())
        return result

    def capture_mixing(h22, *args, **kwargs):
        mixing_values.append(h22.detach().clone())
        return original_mixing(h22, *args, **kwargs)

    monkeypatch.setattr(xhm_module, "_carrier_h22_ratio", capture_helper)
    monkeypatch.setattr(xhm_module, "_spherical_32_from_spheroidal", capture_mixing)

    from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM

    IMRPhenomXHM(modes=((3, 2),))(
        torch.tensor([30.0, 300.0], dtype=torch.float64),
        mass1=35.0,
        mass2=20.0,
        chi1=0.5,
        chi2=-0.3,
        return_unprojected_modes=True,
    )

    assert helper_values
    assert len(helper_values) == len(mixing_values)
    for helper_value, mixing_value in zip(helper_values, mixing_values):
        torch.testing.assert_close(mixing_value, helper_value, rtol=0.0, atol=0.0)


def test_spherical_32_reconstructs_appendix_a_matrix():
    """The eliminated 22 spheroidal entry exactly satisfies Appendix A Eq. (A3)."""
    alpha222 = torch.tensor(0.93 + 0.04j, dtype=torch.complex128)
    alpha232 = torch.tensor(-0.08 + 0.02j, dtype=torch.complex128)
    alpha322 = torch.tensor(0.11 - 0.03j, dtype=torch.complex128)
    alpha332 = torch.tensor(0.97 + 0.01j, dtype=torch.complex128)
    h22_s = torch.tensor(1.2 - 0.4j, dtype=torch.complex128)
    h32_s = torch.tensor(-0.3 + 0.6j, dtype=torch.complex128)
    h22 = alpha222 * h22_s + alpha232 * h32_s
    expected_h32 = alpha322 * h22_s + alpha332 * h32_s

    result = _spherical_32_from_spheroidal(
        h22, h32_s, alpha222, alpha232, alpha322, alpha332
    )
    torch.testing.assert_close(result, expected_h32, rtol=1e-14, atol=1e-14)


def test_published_mixing_fits_have_no_spin_zero_coupling():
    """The author fits reduce to the identity matrix at the Kerr spin-zero limit."""
    chi = torch.tensor(0.0, dtype=torch.float64)
    alpha222 = torch.complex(*XHMParameterFits.mixing_alpha222(chi))
    alpha232 = torch.complex(*XHMParameterFits.mixing_alpha223(chi))
    alpha322 = torch.complex(*XHMParameterFits.mixing_alpha322(chi))
    alpha332 = torch.complex(*XHMParameterFits.mixing_alpha333(chi))
    torch.testing.assert_close(alpha222, torch.tensor(1.0 + 0.0j, dtype=torch.complex128), rtol=0.0, atol=0.0)
    torch.testing.assert_close(alpha232, torch.tensor(0.0 + 0.0j, dtype=torch.complex128), rtol=0.0, atol=0.0)
    torch.testing.assert_close(alpha322, torch.tensor(0.0 + 0.0j, dtype=torch.complex128), rtol=0.0, atol=0.0)
    torch.testing.assert_close(alpha332, torch.tensor(1.0 + 0.0j, dtype=torch.complex128), rtol=0.0, atol=0.0)

    h22 = torch.tensor(0.6 - 0.2j, dtype=torch.complex128)
    h32_s = torch.tensor(-0.1 + 0.4j, dtype=torch.complex128)
    result = _spherical_32_from_spheroidal(h22, h32_s, alpha222, alpha232, alpha322, alpha332)
    torch.testing.assert_close(result, h32_s, rtol=0.0, atol=0.0)
