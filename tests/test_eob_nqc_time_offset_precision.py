"""Clean-room checks for the published NQC time-offset conventions."""

from decimal import Decimal, localcontext

import torch

from torchwave.approximants.eob.nqc import (
    seobnrv4_nqc_time_offset,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)


def test_nqc_time_offset_rounds_each_coefficient_before_evaluation():
    printed = ("716.044155", "-13.087878", "-45.883834", "-2.504992",
               "-0.192775", "19.053803", "-11.543497", "40.318332", "-13.006363")
    points = (("0.24", "0"), ("0.19", "0.37"), ("0.08", "-0.61"))
    with localcontext() as context:
        context.prec = 6
        rounded = tuple(+Decimal(value) for value in printed)
        with localcontext() as high:
            high.prec = 40
            originals = tuple(Decimal(value) for value in printed)
            for nu_text, chi_text in points:
                nu_d, chi_d = Decimal(nu_text), Decimal(chi_text)
                expected_d = (rounded[0]*nu_d**3 + rounded[1]*nu_d**2 + rounded[2]*nu_d + rounded[3]
                              + rounded[4]*chi_d**3*nu_d**2 + rounded[5]*chi_d**3*nu_d
                              + rounded[6]*chi_d**2 + rounded[7]*chi_d*nu_d + rounded[8]*chi_d)
                original_d = (originals[0]*nu_d**3 + originals[1]*nu_d**2 + originals[2]*nu_d + originals[3]
                              + originals[4]*chi_d**3*nu_d**2 + originals[5]*chi_d**3*nu_d
                              + originals[6]*chi_d**2 + originals[7]*chi_d*nu_d + originals[8]*chi_d)
                with localcontext() as final_context:
                    final_context.prec = 6
                    final_rounded_d = +original_d
                nu, chi = torch.tensor(float(nu_d), dtype=torch.float64), torch.tensor(float(chi_d), dtype=torch.float64)
                actual = seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi)
                torch.testing.assert_close(actual, torch.tensor(float(expected_d), dtype=torch.float64), rtol=0.0, atol=1e-13)
                published = seobnrv4_nqc_time_offset(nu, chi)
                torch.testing.assert_close(published, torch.tensor(float(original_d), dtype=torch.float64), rtol=0.0, atol=1e-12)
                # Rounding the final paper polynomial output is a different contract.
                assert abs(float(actual) - float(final_rounded_d)) > 1e-10


def test_nqc_time_offset_compatibility_helper_retains_autograd():
    nu = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
    chi = torch.tensor(-0.43, dtype=torch.float64, requires_grad=True)
    value = seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi)
    grad_nu, grad_chi = torch.autograd.grad(value, (nu, chi))
    expected_nu = 3*716.044*nu.detach()**2 + 2*(-13.0879)*nu.detach() - 45.8838
    expected_nu += 2*(-0.192775)*chi.detach()**3*nu.detach()
    expected_nu += 19.0538*chi.detach()**3 + 40.3183*chi.detach()
    expected_chi = (3*(-0.192775)*chi.detach()**2*nu.detach()**2
                   + 3*19.0538*chi.detach()**2*nu.detach()
                   + 2*(-11.5435)*chi.detach() + 40.3183*nu.detach()
                   - 13.0064)
    torch.testing.assert_close(grad_nu, expected_nu, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(grad_chi, expected_chi, rtol=0.0, atol=1e-12)
