"""Clean-room PN diagnostic factors for the frequency-domain (2,1) mode.

The ``v**4`` coefficient is a direct transcription of Mishra et al.,
arXiv:1601.05588, Eq. (H21) (the paper's Eq. 12b).  This module is used by
diagnostic scripts only; it is not imported by a production approximant.
"""
from __future__ import annotations

import torch


def hhat21_eq12b_v4(v, eta, delta, chi_s, chi_a):
    """Return Eq. (H21), including its displayed complex ``V_1**4`` term."""
    v = torch.as_tensor(v)
    eta = torch.as_tensor(eta, dtype=v.dtype, device=v.device)
    delta = torch.as_tensor(delta, dtype=v.dtype, device=v.device)
    chi_s = torch.as_tensor(chi_s, dtype=v.dtype, device=v.device)
    chi_a = torch.as_tensor(chi_a, dtype=v.dtype, device=v.device)
    c = chi_a * (4771.0 / 1344.0 - 11941.0 * eta / 336.0)
    c = c + delta * chi_s * (4771.0 / 1344.0 - 2549.0 * eta / 336.0)
    c = c + delta * (-torch.pi - 0.5j - 2j * torch.log(v.new_tensor(2.0)))
    return -(torch.sqrt(v.new_tensor(2.0)) / 3.0) * (
        delta * v - 1.5 * (chi_a + delta * chi_s) * v**2
        + delta * (335.0 / 672.0 + 117.0 * eta / 56.0) * v**3 + c * v**4
    )


def hhat21_spa_spin_v4(v, eta, delta, chi_s, chi_a):
    """Diagnostic-only SPA product from Arun et al. 0810.5336.

    Appendix-C Eq. (derivexpansion) has the leading SO part of ``S_3``.
    Multiplying it by the Newtonian ``C_1^(1)`` mode gives the displayed
    legacy 1.5PN-SPA spin contribution at absolute ``v**4``.  This deliberately
    excludes Mishra's newer direct 2PN mode terms.
    """
    v = torch.as_tensor(v)
    eta = torch.as_tensor(eta, dtype=v.dtype, device=v.device)
    delta = torch.as_tensor(delta, dtype=v.dtype, device=v.device)
    chi_s = torch.as_tensor(chi_s, dtype=v.dtype, device=v.device)
    chi_a = torch.as_tensor(chi_a, dtype=v.dtype, device=v.device)
    c = delta * ((113.0 / 24.0 - 19.0 * eta / 6.0) * chi_s + (113.0 / 24.0) * delta * chi_a)
    return -(torch.sqrt(v.new_tensor(2.0)) / 3.0) * c * v**4


def hhat21_nonspin_v4(v, eta, delta, chi_s, chi_a):
    """Eq. (H21)'s displayed nonspinning tail-only V4 increment."""
    v = torch.as_tensor(v)
    delta = torch.as_tensor(delta, dtype=v.dtype, device=v.device)
    return -(torch.sqrt(v.new_tensor(2.0)) / 3.0) * delta * (
        -torch.pi - 0.5j - 2j * torch.log(v.new_tensor(2.0))
    ) * v**4


def hhat21_arun_compat_v4(v, eta, delta, chi_s, chi_a):
    """Diagnostic sign control from the displayed Arun SPA algebra.

    Arun's displayed ``S_3`` and the 2PN time-domain spin contribution give
    ``c_SPA-c_TD`` for the public black-box observations, whereas the printed
    Mishra Eq. (12b) coefficient is ``c_SPA+c_TD``.  The expression below
    keeps the published nonspinning tail and reverses only that source-derived
    TD spin term.  It is a diagnostic control, never a production constant.
    """
    v = torch.as_tensor(v)
    eta = torch.as_tensor(eta, dtype=v.dtype, device=v.device)
    delta = torch.as_tensor(delta, dtype=v.dtype, device=v.device)
    chi_s = torch.as_tensor(chi_s, dtype=v.dtype, device=v.device)
    chi_a = torch.as_tensor(chi_a, dtype=v.dtype, device=v.device)
    c_spa = delta * ((113.0 / 24.0 - 19.0 * eta / 6.0) * chi_s
                     + (113.0 / 24.0) * delta * chi_a)
    c_spa = c_spa - 1.5 * (chi_a + delta * chi_s) * (
        743.0 / 672.0 + 11.0 * eta / 8.0)
    c_td = (0.5 - 205.0 * eta / 14.0) * chi_a + delta * (
        0.5 - 33.0 * eta / 14.0) * chi_s
    c_tail = delta * (-torch.pi - 0.5j - 2j * torch.log(v.new_tensor(2.0)))
    c = c_tail + c_spa - c_td
    return -(torch.sqrt(v.new_tensor(2.0)) / 3.0) * (
        delta * v - 1.5 * (chi_a + delta * chi_s) * v**2
        + delta * (335.0 / 672.0 + 117.0 * eta / 56.0) * v**3 + c * v**4
    )
