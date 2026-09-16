# SPDX-License-Identifier: MIT
"""Independent PN Fourier-mode amplitudes from arXiv:2001.10914v2.

The expressions below are the published Appendix E leading and explicitly
displayed low-order terms, with the common SPA factor removed.  This module is
diagnostic only; it contains no XHM calibration coefficients or runtime calls.
"""
from __future__ import annotations
import math
import torch

PRINTED_PAPER_VERSION = "printed-paper"
COMPATIBILITY_VERSION = "compatibility"


def _broadcast_hhat_inputs(v, eta, delta, chi_s, chi_a):
    return torch.broadcast_tensors(
        *[
            torch.as_tensor(x, dtype=torch.float64)
            for x in (v, eta, delta, chi_s, chi_a)
        ]
    )


def hhat_lm_printed(mode, v, eta, delta, chi_s, chi_a):
    """Return complex ``Hhat_lm``; v is V_m=(2*pi*M*f/m)**(1/3).

    Provenance: arXiv:2001.10914v2 Appendix E, E10 (21), E11 (33), E12
    (32), E14 (44), and E15 (43).  Those five modes are transcribed through
    every displayed v^6 term.  The (2,2) value is only the leading reference
    normalization, namely the displayed ``1``; this helper does not claim to
    implement the full E9 series.
    """
    v, eta, delta, cs, ca = torch.broadcast_tensors(*[torch.as_tensor(x, dtype=torch.float64) for x in (v, eta, delta, chi_s, chi_a)])
    z = torch.zeros_like(v)
    i = torch.complex(z, torch.ones_like(z))
    if mode == (2, 1):
        b = (v*delta + v**2*(-1.5*delta*cs-1.5*ca)
             + v**3*(117*delta*eta/56+335*delta/672)
             + v**4*(-965*delta*eta*cs/336+3427*delta*cs/1344
                      -math.pi*delta-0.5j*delta-0.5j*delta*math.log(16)
                      -2101*eta*ca/336+3427*ca/1344)
             + v**5*(21365*delta*eta**2/8064 + 10*delta*eta*ca**2
                      +39*delta*eta*cs**2/8 -36529*delta*eta/12544
                      -307*delta*ca**2/32 -307*delta*cs**2/32
                      +3*math.pi*delta*cs -964357*delta/8128512
                      +213*eta*ca*cs/4 -307*ca*cs/16 +3*math.pi*ca)
             + v**6*(-547*delta*eta**2*cs/768 -15*delta*eta*ca**2*cs
                      -3*delta*eta*cs**3/16 -7049629*delta*eta*cs/225792
                      +417*math.pi*delta*eta/112 -1489j*delta*eta/112
                      -89j*delta*eta*math.log(2)/28 +729*delta*ca**2*cs/64
                      +243*delta*cs**3/64 +143063173*delta*cs/5419008
                      -2455*math.pi*delta/1344 -335j*delta/1344
                      -335j*delta*math.log(2)/336 +42617*eta**2*ca/1792
                      -15*eta*ca**3 -489*eta*ca*cs**2/16
                      -22758317*eta*ca/225792 +243*ca**3/64
                      +729*ca*cs**2/64 +143063173*ca/5419008))
        return i*math.sqrt(2)/3*b
    if mode == (3, 3):
        b=v*delta + v**3*delta*(27*eta/8-1945/672)
        b=b+v**4*(-2*delta*eta*cs/3+65*delta*cs/24+math.pi*delta
                  -21j*delta/5+6j*delta*math.log(3/2)-28*eta*ca/3+65*ca/24)
        b=b+v**5*(420389*delta*eta**2/63360+10*delta*eta*ca**2
                  +delta*eta*cs**2/8-11758073*delta*eta/887040
                  -81*delta*ca**2/32-81*delta*cs**2/32
                  -1077664867*delta/447068160+81*eta*ca*cs/4
                  -81*ca*cs/16)
        b=b+v**6*(-67*delta*eta**2*cs/24-58745*delta*eta*cs/4032
                  +131*math.pi*delta*eta/16-440957j*delta*eta/9720
                  +69j*delta*eta*math.log(3/2)/4+163021*delta*cs/16128
                  -5675*math.pi*delta/1344+389j*delta/32
                  -1945j*delta*math.log(3/2)/112-137*eta**2*ca/24
                  -148501*eta*ca/4032+163021*ca/16128)
        return -0.75*i*math.sqrt(5/7)*b
    if mode == (3, 2):
        b = v**2*(1-3*eta) + 4*eta*cs*v**3
        b = b + v**4*(-589*eta**2/72+12325*eta/2016-10471/10080)
        b = b + v**5*(eta*(113*delta*ca/8+1081*cs/84-66j/5)
                      +(-113*delta*ca-113*cs+72j)/24-15*eta**2*cs)
        b = b + v**6*(eta*(-1633*delta*ca*cs/48-563*ca**2/32
                      -2549*cs**2/96+8*math.pi*cs-8689883/149022720)
                      +81*delta*ca*cs/16+837223*eta**3/63360
                      +eta**2*(30*ca**2+313*cs**2/24-78584047/2661120)
                      +81*ca**2/32+81*cs**2/32+824173699/447068160)
        return math.sqrt(5/7)/3*b
    if mode == (4, 4):
        b = v**2*(3*eta-1) + v**4*(1063*eta**2/88-128221*eta/7392+158383/36960)
        b = b + v**5*(math.pi*(2-6*eta)+(-eta*(1695*delta*ca+2075*cs
                      -3579j+2880j*math.log(2))+565*delta*ca
                      +1140*eta**2*cs+565*cs-1008j+960j*math.log(2))/120)
        b = b + v**6*(eta*(243*delta*ca*cs/16+563*ca**2/32+247*cs**2/32
                      -22580029007/880588800)-81*delta*ca*cs/16
                      -7606537*eta**3/274560
                      +eta**2*(-30*ca**2-3*cs**2/8+901461137/11531520)
                      -81*ca**2/32-81*cs**2/32+7888301437/29059430400)
        return 4/9*math.sqrt(10/7)*b
    if mode == (4, 3):
        b = v**3*(2*delta*eta-delta) + v**4*(2.5*eta*ca-2.5*delta*eta*cs)
        b = b + v**5*(887*delta*eta**2/132-10795*delta*eta/1232+18035*delta/7392)
        b = b + v**6*(-469*delta*eta**2*cs/48+4399*delta*eta*cs/448
                      +2*math.pi*delta*eta-16301j*delta*eta/810
                      # The source prints 12 i delta eta (3/2), with no log
                      # on this term.  Transcribe it literally as 18 i delta
                      # eta; the adjacent -6 i delta log(3/2) is separate.
                      # This possible paper typo is left unresolved here.
                      +18j*delta*eta-113*delta*cs/24
                      -math.pi*delta+32j*delta/5-6j*delta*math.log(3/2)
                      -1643*eta**2*ca/48+41683*eta*ca/1344-113*ca/24)
        return 0.75*i*math.sqrt(3/35)*b
    if mode == (2, 2):
        return torch.ones_like(v, dtype=torch.complex128)
    raise ValueError(f"unsupported published Appendix E mode {mode}")

def hhat_lm_compatibility(mode, v, eta, delta, chi_s, chi_a):
    """Return the XHM compatibility polynomial.

    The compatibility convention changes only the printed low-order terms
    that were signed oppositely by the earlier public provider: ``v**2`` and
    ``v**4`` for 44, and ``v**2``, ``v**3`` and ``v**4`` for 32.  Every other
    mode and every higher-order term delegates to the explicitly retained
    printed-paper polynomial.
    """
    v, eta, delta, cs, ca = _broadcast_hhat_inputs(v, eta, delta, chi_s, chi_a)
    if mode not in ((3, 2), (4, 4)):
        return hhat_lm_printed(mode, v, eta, delta, cs, ca)

    if mode == (3, 2):
        # The v**2--v**4 terms are the sign-reversed compatibility terms;
        # v**5 and v**6 retain the printed Appendix-E expressions.
        b = -v**2 * (1 - 3 * eta) - 4 * eta * cs * v**3
        b = b - v**4 * (-589 * eta**2 / 72 + 12325 * eta / 2016 - 10471 / 10080)
        b = b + v**5 * (
            eta * (113 * delta * ca / 8 + 1081 * cs / 84 - 66j / 5)
            + (-113 * delta * ca - 113 * cs + 72j) / 24
            - 15 * eta**2 * cs
        )
        b = b + v**6 * (
            eta * (
                -1633 * delta * ca * cs / 48
                -563 * ca**2 / 32
                -2549 * cs**2 / 96
                +8 * math.pi * cs
                -8689883 / 149022720
            )
            +81 * delta * ca * cs / 16
            +837223 * eta**3 / 63360
            +eta**2 * (30 * ca**2 + 313 * cs**2 / 24 - 78584047 / 2661120)
            +81 * ca**2 / 32
            +81 * cs**2 / 32
            +824173699 / 447068160
        )
        return math.sqrt(5 / 7) / 3 * b

    # (4,4): reverse only the displayed v**2 and v**4 terms.  The printed
    # v**5 and v**6 terms are copied through the independent algebraic
    # branch below, so their signs remain unchanged.
    b = -v**2 * (3 * eta - 1)
    b = b - v**4 * (1063 * eta**2 / 88 - 128221 * eta / 7392 + 158383 / 36960)
    b = b + v**5 * (
        math.pi * (2 - 6 * eta)
        + (
            -eta * (1695 * delta * ca + 2075 * cs - 3579j + 2880j * math.log(2))
            +565 * delta * ca
            +1140 * eta**2 * cs
            +565 * cs
            -1008j
            +960j * math.log(2)
        )
        / 120
    )
    b = b + v**6 * (
        eta * (
            243 * delta * ca * cs / 16
            +563 * ca**2 / 32
            +247 * cs**2 / 32
            -22580029007 / 880588800
        )
        -81 * delta * ca * cs / 16
        -7606537 * eta**3 / 274560
        +eta**2 * (-30 * ca**2 - 3 * cs**2 / 8 + 901461137 / 11531520)
        -81 * ca**2 / 32
        -81 * cs**2 / 32
        +7888301437 / 29059430400
    )
    return 4 / 9 * math.sqrt(10 / 7) * b


def hhat_lm(mode, v, eta, delta, chi_s, chi_a, *, version=PRINTED_PAPER_VERSION):
    """Dispatch to the explicitly labelled printed or compatibility form."""
    if version in (PRINTED_PAPER_VERSION, "printed", "paper"):
        return hhat_lm_printed(mode, v, eta, delta, chi_s, chi_a)
    if version in (COMPATIBILITY_VERSION, "compat"):
        return hhat_lm_compatibility(mode, v, eta, delta, chi_s, chi_a)
    raise ValueError(
        f"version must be {PRINTED_PAPER_VERSION!r} or {COMPATIBILITY_VERSION!r}"
    )


def normalized_pn_amplitude(
    mode, frequency, total_mass_seconds, m1, m2, chi1, chi2,
    *, version=PRINTED_PAPER_VERSION,
):
    """Compute H=|A_SPA|/A22^0 using Eq. (spaamp_2) and Appendix E."""
    f = torch.as_tensor(frequency, dtype=torch.float64)
    eta = torch.as_tensor(m1*m2/(m1+m2)**2, dtype=torch.float64)
    delta = torch.as_tensor((m1-m2)/(m1+m2), dtype=torch.float64)
    cs = torch.as_tensor((chi1+chi2)/2, dtype=torch.float64)
    ca = torch.as_tensor((chi1-chi2)/2, dtype=torch.float64)
    mm = mode[1]
    v = (2*math.pi*(f*total_mass_seconds)/mm).pow(1/3)
    return hhat_lm(mode, v, eta, delta, cs, ca, version=version).abs() * (mm/2)**(7/6)


def normalized_pn_amplitude_compatibility(
    mode, frequency, total_mass_seconds, m1, m2, chi1, chi2
):
    """Compatibility-normalized PN amplitude used by the old XHM provider."""
    return normalized_pn_amplitude(
        mode,
        frequency,
        total_mass_seconds,
        m1,
        m2,
        chi1,
        chi2,
        version=COMPATIBILITY_VERSION,
    )

def collocation_coefficients(pn_values, collocation_values, f_ins,
                             active_terms=None, retained_nodes=None):
    """Solve Eq. (insp_amp) pseudo-PN coefficients with documented vetoes.

    ``pn_values`` and ``collocation_values`` are values at
    ``[.5,.75,1]*f_ins``.  ``retained_nodes`` explicitly selects the nodes
    retained after a paper-described veto; removed highest-order coefficients
    are returned as zero.  The full three-coefficient vector is always
    returned.  This is the linear system described after Eq. (28), and does
    not select values from an oracle.
    """
    pn=torch.as_tensor(pn_values,dtype=torch.float64)
    y=torch.as_tensor(collocation_values,dtype=torch.float64)-pn
    if active_terms is None:
        active_terms = 3 if retained_nodes is None else len(retained_nodes)
    if active_terms not in (0,1,2,3):
        raise ValueError("active_terms must be 0, 1, 2, or 3")
    if retained_nodes is None:
        retained_nodes = tuple(range(active_terms))
    else:
        retained_nodes = tuple(retained_nodes)
    if len(retained_nodes) != active_terms or len(set(retained_nodes)) != len(retained_nodes):
        raise ValueError("retained_nodes must contain active_terms distinct nodes")
    if any(node not in (0, 1, 2) for node in retained_nodes):
        raise ValueError("retained_nodes entries must be 0, 1, or 2")
    if active_terms == 0:
        return torch.zeros(y.shape[:-1]+(3,),dtype=y.dtype,device=y.device)
    x=torch.tensor([.5,.75,1.],dtype=torch.float64,device=y.device)
    powers=(7/3,8/3,3)
    # The paper removes highest powers first.  For fewer than three nodes,
    # callers provide the corresponding leading rows; solve supports (...,N).
    if y.shape[-1] != 3:
        raise ValueError("collocation values must contain all three nodes")
    mat=torch.stack(tuple(x[list(retained_nodes)]**p for p in powers[:active_terms]),dim=1)
    rhs=y[..., list(retained_nodes)]
    solved=torch.linalg.solve(mat,rhs.unsqueeze(-1))[...,0]
    result=torch.zeros(y.shape[:-1]+(3,),dtype=y.dtype,device=y.device)
    result[..., :active_terms] = solved
    return result
