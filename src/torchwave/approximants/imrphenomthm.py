"""Native aligned-spin IMRPhenomTHM (21, 33, 44 and 55).

The ansatzes follow Estellés et al., Phys. Rev. D 105, 084039 (2022), and the
MIT licensed ``phenomxpy`` reference at commit
``4121fafa0484f3db2a7352f4eea697faaaf29f65``.  This module deliberately
contains tensor algebra only; the numerical calibration polynomials live in
``diffgw_core.phenomt_fits``.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC, PI
from .imrphenomt import IMRPhenomT

_POSITIVE = ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5))
_OFF = {22: 0.0, 21: math.pi / 2, 33: -math.pi / 2, 44: math.pi, 55: math.pi / 2}


def _validate_modes(mode_array):
    sel = tuple(tuple(pair) for pair in mode_array)
    if not sel:
        raise ValueError("mode selection cannot be empty")
    allowed = {(l, m) for l, m in _POSITIVE} | {(l, -m) for l, m in _POSITIVE}
    if any(pair not in allowed for pair in sel):
        raise ValueError(f"unsupported THM mode selection {sel}")
    return sel


def _fit(name, *args):
    from diffgw_core import phenomt_fits as f
    fn = getattr(f, name, None) or getattr(f, "_" + name, None)
    if fn is None:
        raise AttributeError(name)
    return fn(*args)


def _mode_pn(mode, eta, delta, chi1, chi2):
    """Complex PN amplitude coefficients from THM Eq. (9)."""
    cs, ca = (chi1 + chi2) / 2, (chi1 - chi2) / 2
    m1, m2 = (1 + delta) / 2, (1 - delta) / 2
    sc = m1.square() * chi1 + m2.square() * chi2
    sig = m2 * chi2 - m1 * chi1
    e2, e3 = eta.square(), eta**3
    z = torch.zeros_like(eta)
    if mode == 22:
        s0 = m1 * chi1 + m2 * chi2
        re = (z + 1, z, z + (-107 / 42 + 55 * eta / 42), -107 * PI / 21 + 34 * eta * PI / 21, -2173 / 1512 - 1069 * eta / 216 + 2047 * e2 / 1512 + s0.square(), -2173 * PI / 756 - 2495 * eta * PI / 378 + 40 * e2 * PI / 27, 27027409 / 646800 - 278185 * eta / 33264 - 20261 * e2 / 2772 + 114635 * e3 / 99792 - 856 * 0.5772156649015329 / 105 + 2 * PI**2 / 3 + 41 * eta * PI**2 / 96, z, -428 / 105)
        im = (z, z, z, -24 * eta, 428 * PI / 105, 14333 * eta / 162 - 4066 * e2 / 945)
        fac = 2 * eta * math.sqrt(16 * PI / 5)
    elif mode == 21:
        re = (z, delta / 3, -ca / 2 - cs * delta / 2, -17 * delta / 84 + 5 * delta * eta / 21, delta * PI / 3 - 43 * delta * sc / 21 - 79 * sig / 42 + 139 * eta * sig / 42, -43 * delta / 378 - 509 * delta * eta / 378 + 79 * delta * e2 / 504, -17 * delta * PI / 84 + delta * eta * PI / 14, z, z)
        im = (z, z, z, -delta / 6 - delta * math.log(16) / 6, z, 17 * delta / 168 - 353 * delta * eta / 84 + 17 * delta * math.log(16) / 168 - delta * eta * math.log(4096) / 84, z)
        fac = 2 * eta * math.sqrt(16 * PI / 5)
    elif mode == 33:
        q = math.sqrt(15 / 14)
        re = (z, 3 * q * delta / 4, z, -3 * q * delta + 1.5 * q * delta * eta, 9 * q * delta * PI / 4 - 3 * math.sqrt(105 / 2) * delta * sc / 8 - 9 * q * sig / 8 + 27 * q * eta * sig / 8, 369 * math.sqrt(3 / 70) * delta / 88 - 919 * math.sqrt(3 / 70) * delta * eta / 22 + 887 * math.sqrt(3 / 70) * delta * e2 / 88, z, z, z)
        im = (z, z, z, -9 * math.sqrt(21 / 10) * delta / 4 + 9 * q * delta * math.log(1.5) / 2, z, z, z)
        fac = 2 * eta * math.sqrt(16 * PI / 5)
    elif mode == 44:
        q = math.sqrt(5 / 7)
        re = (z, z, 8 * q / 9 - 8 * q * eta / 3, z, -2372 / (99 * math.sqrt(35)) + 5092 * q * eta / 297 - 100 * math.sqrt(35) * e2 / 99, 32 * q * PI / 9 - 32 * q * eta * PI / 3, 1068671 / (45045 * math.sqrt(35)) - 1088119 * eta / (6435 * math.sqrt(35)) + 293758 * e2 / (1053 * math.sqrt(35)) - 226097 * e3 / (3861 * math.sqrt(35)), z, z)
        # The 44 imaginary 2.5PN term is the fifth coefficient (the source
        # stores an explicit zero 2PN slot before it).
        im = (z, z, z, z, -16 * math.sqrt(7 / 5) / 3 + 1193 * eta / (9 * math.sqrt(35)) + 64 * q * math.log(2) / 9 - 64 * q * eta * math.log(2) / 3, z, z)
        fac = 2 * eta * math.sqrt(16 * PI / 5)
    elif mode == 55:
        q = math.sqrt(66)
        re = (z, z, z, 625 * delta / (96 * q) - 625 * delta * eta / (48 * q), z, -164375 * delta / (3744 * q) + 26875 * delta * eta / (234 * q) - 2500 * math.sqrt(2 / 33) * delta * e2 / 117, 3125 * delta * PI / (96 * q) - 3125 * delta * eta * PI / (48 * q), z, z)
        im = (z, z, z, z, z, -113125 * delta / (1344 * q) + 17639 * delta * eta / (80 * q) + 3125 * delta * math.log(2.5) / (48 * q) - 3125 * delta * eta * math.log(2.5) / (24 * q), z)
        fac = 2 * eta * math.sqrt(16 * PI / 5)
    else:
        raise ValueError(f"unsupported THM mode {mode}")
    return tuple(re), tuple(im), fac


def _pn_amp(x, coeff, pseudo):
    re, im, fac = coeff
    h = torch.zeros_like(x)
    j = torch.zeros_like(x)
    for c, p in zip(re, (0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 3)):
        h = h + c[..., None] * x**p
    h = h + re[8][..., None] * torch.log(16 * x) * x**3
    for c, p in zip(im, (.5, 1, 1.5, 2, 2.5, 3, 3.5)):
        j = j + c[..., None] * x**p
    h = h + pseudo[..., 0, None] * x**4 + pseudo[..., 1, None] * x**4.5 + pseudo[..., 2, None] * x**5
    return fac[..., None] * x * torch.complex(h, j)


def _pn_amp_abs_derivative(x, coeff, pseudo, omega, domega):
    """Analytic d|h_PN|/dt, including the calibrated pseudo-PN terms."""
    re, im, fac = coeff
    rp = torch.zeros_like(x)
    ip = torch.zeros_like(x)
    r = torch.zeros_like(x)
    q = torch.zeros_like(x)
    for c, p in zip(re[:8], (0, .5, 1, 1.5, 2, 2.5, 3, 3.5)):
        r = r + c * x**p
        if p:
            rp = rp + p * c * x ** (p - 1)
    r = r + re[8] * torch.log(16 * x) * x**3
    rp = rp + re[8] * x**2 * (1 + 3 * torch.log(16 * x))
    for c, p in zip(im, (.5, 1, 1.5, 2, 2.5, 3, 3.5)):
        q = q + c * x**p
        ip = ip + p * c * x ** (p - 1)
    for c, p in zip(pseudo.unbind(-1), (4, 4.5, 5)):
        r = r + c * x**p
        rp = rp + p * c * x ** (p - 1)
    dr = r + x * rp
    di = q + x * ip
    # x=(Omega_22/2)^(2/3), so dx/dt = (2 x / 3 Omega_22) dOmega_22/dt.
    dxdt = (2 * x / (3 * omega.clamp_min(1e-30))) * domega
    return fac * (r * dr + q * di) / torch.sqrt(r.square() + q.square() + 1e-60) * dxdt


def _pn_amp_phase_derivative(x, coeff, pseudo, omega, domega):
    """Analytic d arg(h_PN)/dt used by the HM phase cut correction."""
    re, im, fac = coeff
    rp = torch.zeros_like(x)
    ip = torch.zeros_like(x)
    r = torch.zeros_like(x)
    q = torch.zeros_like(x)
    for c, p in zip(re[:8], (0, .5, 1, 1.5, 2, 2.5, 3, 3.5)):
        r = r + c * x**p
        if p:
            rp = rp + p * c * x ** (p - 1)
    r = r + re[8] * torch.log(16 * x) * x**3
    rp = rp + re[8] * x**2 * (1 + 3 * torch.log(16 * x))
    for c, p in zip(im, (.5, 1, 1.5, 2, 2.5, 3, 3.5)):
        q = q + c * x**p
        ip = ip + p * c * x ** (p - 1)
    for c, p in zip(pseudo.unbind(-1), (4, 4.5, 5)):
        r = r + c * x**p
        rp = rp + p * c * x ** (p - 1)
    dxdt = (2 * x / (3 * torch.clamp(omega, min=1e-30))) * domega
    return (ip * r - rp * q) / torch.clamp(r.square() + q.square(), min=1e-30) * dxdt


def _basis(t, alpha, derivative=False):
    x = alpha[..., None] * t
    s1, s2 = 1 / torch.cosh(x), 1 / torch.cosh(2 * x)
    if derivative:
        return torch.stack((torch.zeros_like(t), -alpha[..., None] * s1 * torch.tanh(x), -2 / 7 * alpha[..., None] * torch.tanh(2 * x) * s2 ** (1 / 7), 2 * t), -1)
    return torch.stack((torch.ones_like(t), s1, s2 ** (1 / 7), t.square()), -1)


class IMRPhenomTHM(nn.Module):
    """Aligned-spin dominant and subdominant time-domain modes."""
    supported_modes = _POSITIVE

    def __init__(self, mode_array: Iterable[tuple[int, int]] | None = None):
        super().__init__()
        self.mode_array = _validate_modes(mode_array) if mode_array is not None else _POSITIVE + tuple((l, -m) for l, m in _POSITIVE)
        self._tmodel = IMRPhenomT()

    def _positive(self, u, state22, mode, eta, delta, S, dchi, chi1, chi2, final_spin=None):
        l, m = mode
        om22, ph22, a22 = self._tmodel._evaluate(u, state22)
        if mode == (2, 2):
            from ._phenomt_amplitude import amplitude
            return amplitude(u, a22, lambda tt: self._tmodel._evaluate(tt, state22)[0]), ph22, om22
        # The published HM inspiral phase is m/2 times the cached 22 phase.
        phase = m / 2 * ph22 - _OFF[10 * l + m]
        om = m / 2 * om22
        # THM's PN amplitude uses the orbital (22) frequency.
        x = torch.clamp(om22 / 2, min=1e-30) ** (2 / 3)
        coeff = _mode_pn(10 * l + m, eta, delta, chi1, chi2)
        cp = torch.stack([_fit("_IMRPhenomT_Inspiral_Amp_CP", eta, S, dchi, delta, 10 * l + m, i) for i in (1, 2, 3)], -1)
        # Collocation times are fixed in total-mass units; evaluate from 22.
        uc = eta[..., None] * 0 + torch.as_tensor((-2000., -250., -150.), dtype=eta.dtype, device=eta.device)
        # The PN amplitude uses x=(omega_22/2)^(2/3), even for m != 2.
        oc = self._tmodel._evaluate(uc, state22)[0]
        xx = torch.clamp(oc / 2, min=1e-30) ** (2 / 3)
        b0 = _pn_amp(xx, coeff, torch.zeros(eta.shape + (3,), dtype=eta.dtype, device=eta.device)).real
        A = torch.stack((xx**4, xx**4.5, xx**5), -1)
        den = coeff[2]
        den = torch.where(den.abs() < 1e-30, torch.where(den < 0, torch.full_like(den, -1e-30), torch.full_like(den, 1e-30)), den)
        pseudo = torch.linalg.solve(A, ((cp - b0) / (den[..., None] * xx)).unsqueeze(-1))[..., 0]
        # The augmented spin supplied by precession changes the QNM rates in
        # the ringdown only.  Keep the aligned remnant for the merger
        # collocation system so its inspiral/merger portion remains identical
        # to the aligned model; amplitude and phase are then joined at their
        # respective fitted amplitude and frequency cuts.
        af_merger = _fit("IMRPhenomX_FinalSpin2017", eta, chi1, chi2, delta)
        af_rd = af_merger if final_spin is None else final_spin
        mf = _fit("IMRPhenomX_FinalMass2017", eta, chi1, chi2, delta)
        fd = _fit("IMRPhenomT_fdampfit", af_merger, 10 * l + m) / mf
        alpha = 2 * PI * fd
        fd_rd = _fit("IMRPhenomT_fdampfit", af_rd, 10 * l + m) / mf
        alpha_rd = 2 * PI * fd_rd
        peak = _fit("_IMRPhenomT_PeakAmp", eta, S, dchi, delta, 10 * l + m)
        mid = _fit("_IMRPhenomT_Intermediate_Amp_CP1", eta, S, dchi, delta, 10 * l + m)
        c3 = _fit("_IMRPhenomT_Ringdown_Amp_C3", eta, S, dchi, 10 * l + m)
        tshift = _fit("_IMRPhenomT_tshift", eta, S, dchi, 10 * l + m)
        c2 = (2 * PI * _fit("IMRPhenomT_fdampn2fit", af_merger, 10 * l + m) / mf - alpha) / 2
        c2 = torch.where(c2 > torch.abs(-.5 * alpha / torch.tanh(c3)), -.5 * alpha / torch.tanh(c3), c2)
        c2_rd = (2 * PI * _fit("IMRPhenomT_fdampn2fit", af_rd, 10 * l + m) / mf - alpha_rd) / 2
        c2_rd = torch.where(c2_rd > torch.abs(-.5 * alpha_rd / torch.tanh(c3)), -.5 * alpha_rd / torch.tanh(c3), c2_rd)
        c1, c4 = peak * alpha_rd * torch.cosh(c3).square() / c2_rd, peak - peak * alpha_rd * torch.cosh(c3).square() / c2_rd * torch.tanh(c3)
        mt = eta[..., None] * 0 + torch.as_tensor((-150., -25., 0.), dtype=eta.dtype, device=eta.device)
        amp_cut = _pn_amp(xx[..., -1:], coeff, pseudo)[..., 0]
        ic = torch.sign(amp_cut.real) * torch.abs(amp_cut)
        A4 = torch.cat((_basis(mt - tshift[..., None], alpha), _basis((mt - tshift[..., None])[..., :1], alpha, True)), -2)
        # The published matching system fixes the peak row's quadratic basis
        # entry to zero (the ansatz itself still evaluates (t-tshift)^2).
        row2 = torch.cat((torch.ones_like(A4[..., 2:3, :3]), torch.zeros_like(A4[..., 2:3, 3:])), -1)
        A4 = torch.cat((A4[..., :2, :], row2, A4[..., 3:, :]), -2)
        omega_cut = self._tmodel._evaluate(torch.as_tensor(-150., dtype=u.dtype, device=u.device), state22)[0][..., 0]
        domega_cut = self._tmodel._domega_full(torch.as_tensor(-150., dtype=u.dtype, device=u.device), state22)[..., 0]
        x_cut = torch.clamp(omega_cut / 2, min=1e-30) ** (2 / 3)
        # The source ansatz is signed by Re(A_PN) before matching its slope.
        deriv = torch.sign(amp_cut.real) * _pn_amp_abs_derivative(x_cut, coeff, pseudo, omega_cut, domega_cut)
        rhs = torch.stack((ic, mid, peak, deriv), -1)
        mc = torch.linalg.solve(A4, rhs[..., None])[..., 0]
        # The amplitude ansatz switches to ringdown at tshift (the fitted
        # ringdown cut), while the phase/frequency switch is at zero.
        tm = torch.minimum(torch.maximum(u, torch.as_tensor(-150., dtype=u.dtype, device=u.device)), tshift[..., None])
        tr = torch.maximum(u, tshift[..., None])
        am = torch.sum(_basis(tm - tshift[..., None], alpha) * mc[..., None, :], -1)
        tr = tr - tshift[..., None]
        ar = torch.exp(-alpha_rd[..., None] * tr) * (c1[..., None] * torch.tanh(c2_rd[..., None] * tr + c3[..., None]) + c4[..., None])
        amp_complex = _pn_amp(torch.clamp(x, min=1e-30), coeff, pseudo)
        # Keep every branch complex so autograd never sends a complex cotangent
        # into the real merger/RD amplitudes (a PyTorch dtype mismatch).
        am_complex = torch.complex(am, torch.zeros_like(am))
        ar_complex = torch.complex(ar, torch.zeros_like(ar))
        amp = torch.where(u < -150, amp_complex, torch.where(u < tshift[..., None], am_complex, ar_complex))

        # Independent HM merger/ringdown frequency ansatz and its analytic
        # primitive (THM Eqs. 15, 25, 28-31), joined to m/2 times the 22 phase.
        op = _fit("_IMRPhenomT_PeakFrequency", eta, S, dchi, delta, 10 * l + m)
        c2f = _fit("_IMRPhenomT_RD_Freq_C2", eta, S, dchi, delta, 10 * l + m)
        c3f = _fit("_IMRPhenomT_RD_Freq_C3", eta, S, dchi, delta, 10 * l + m)
        wr = 2 * PI * _fit("IMRPhenomT_fringfit", af_merger, 10 * l + m) / mf
        aa = 2 * PI * _fit("IMRPhenomT_fdampfit", af_merger, 10 * l + m) / mf
        c1f = (1 + c3f) * (wr - op) / (c2f * c3f)
        wr_rd = 2 * PI * _fit("IMRPhenomT_fringfit", af_rd, 10 * l + m) / mf
        c1f_rd = (1 + c3f) * (wr_rd - op) / (c2f * c3f)
        cut = torch.as_tensor(-150., dtype=u.dtype, device=u.device)
        tcp = torch.as_tensor(-25., dtype=u.dtype, device=u.device)
        w22cut = self._tmodel._evaluate(cut, state22)[0][..., 0]
        dw22cut = self._tmodel._domega_full(cut, state22)[..., 0]
        x_cut = torch.clamp(w22cut / 2, min=1e-30) ** (2 / 3)
        omega_cut_amp = -_pn_amp_phase_derivative(x_cut, coeff, pseudo, w22cut, dw22cut)
        wcut = m / 2 * w22cut + omega_cut_amp
        dwcut = m / 2 * dw22cut
        dwp = -c1f * c2f.square() * c3f / (1 + c3f).square() / wr
        ac, am = torch.asinh(aa * cut), torch.asinh(aa * tcp)
        wtcp = _fit("_IMRPhenomT_Intermediate_Freq_CP1", eta, S, dchi, delta, 10 * l + m)
        b = torch.stack((1 - wcut / wr - (1 - op / wr) - dwp / aa * ac,
                         1 - wtcp / wr - (1 - op / wr) - dwp / aa * am,
                         -dwcut / wr - dwp / torch.sqrt(1 + (aa * cut).square())), -1)
        matf = torch.stack((torch.stack((ac.square(), ac**3, ac**4), -1), torch.stack((am.square(), am**3, am**4), -1), torch.stack((2*aa*ac/torch.sqrt(1+(aa*cut).square()), 3*aa*ac.square()/torch.sqrt(1+(aa*cut).square()), 4*aa*ac**3/torch.sqrt(1+(aa*cut).square())), -1)), -2)
        cf = torch.linalg.solve(matf, b[..., None])[..., 0]
        tm = torch.clamp(u, -150., 0.)
        tr = torch.clamp(u, min=0.)
        xxm = torch.asinh(aa[..., None] * tm)
        wm = wr[..., None] * (1 - (1 - op[..., None]/wr[..., None] + xxm * (dwp[..., None]/aa[..., None] + xxm * (cf[...,0,None] + xxm*(cf[...,1,None] + xxm*cf[...,2,None])))))
        wrg = wr_rd[..., None] - c1f_rd[..., None]*c2f[..., None]*c3f[..., None]*torch.exp(-c2f[..., None]*tr)/(1+c3f[..., None]*torch.exp(-c2f[..., None]*tr))
        om = torch.where(u < -150, m/2*om22, torch.where(u < 0, wm, wrg))
        # The join phase is evaluated at the scalar cut, rather than borrowed
        # from whichever time samples the caller happened to request.
        phcut22 = self._tmodel._evaluate(cut, state22)[1][..., 0]
        # The complex PN amplitude shifts both the HM cut frequency and phase.
        # atan2 has an undefined derivative at (0, 0).  Odd modes vanish at
        # equal mass/equal spin, so evaluate the inactive branch at a safe
        # unit real amplitude before masking its phase.
        amp_nonzero = torch.abs(amp_cut) > 1e-30
        amp_re = torch.where(amp_nonzero, amp_cut.real, torch.ones_like(amp_cut.real))
        amp_im = torch.where(amp_nonzero, amp_cut.imag, torch.zeros_like(amp_cut.imag))
        phicut = torch.atan2(amp_im, amp_re) + torch.where(amp_re < 0, torch.as_tensor(PI, dtype=u.dtype, device=u.device), torch.zeros_like(amp_re))
        phicut = torch.where(amp_nonzero, phicut, torch.zeros_like(phicut))
        rawcut = m / 2 * phcut22 - phicut - _OFF[10 * l + m]
        from .imrphenomt import IMRPhenomT as _T
        inter = _T._phase_inter_raw(tm, aa[..., None], wr[..., None], op[..., None], dwp[..., None], cf[..., None, :])
        inter0 = _T._phase_inter_raw(torch.zeros_like(eta), aa, wr, op, dwp, cf)
        offm = rawcut - _T._phase_inter_raw(cut, aa, wr, op, dwp, cf)
        rphase = c1f_rd[..., None] * torch.log((1 + c3f[..., None]*torch.exp(-c2f[..., None]*tr))/(1+c3f[..., None])) + wr_rd[..., None]*tr + inter0[..., None] + offm[..., None]
        phase = torch.where(u < -150, phase, torch.where(u < 0, inter + offm[..., None], rphase))
        odd_zero = (torch.abs(delta) < 1e-14) & (torch.abs(chi1 - chi2) < 1e-14)
        odd_zero = odd_zero & (m % 2 == 1)
        amp = torch.where(odd_zero[..., None], torch.zeros_like(amp), amp)
        phase = torch.where(odd_zero[..., None], torch.zeros_like(phase), phase)
        om = torch.where(odd_zero[..., None], torch.zeros_like(om), om)
        return amp, phase, om

    def modes(self, times, mass1, mass2, chi1=0., chi2=0., distance=100., phic=0., tc=0., f_ref=20., mode_array=None, final_spin=None, modes=None):
        t = torch.as_tensor(times, dtype=times.dtype if isinstance(times, torch.Tensor) and times.is_floating_point() else torch.float64, device=times.device if isinstance(times, torch.Tensor) else None)
        vals = [torch.as_tensor(v, dtype=t.dtype, device=t.device) for v in (mass1, mass2, chi1, chi2, distance, phic, tc)]
        fref = torch.as_tensor(f_ref, dtype=t.dtype, device=t.device)
        m1, m2, c1, c2, dist, pc, tcenter, fref = torch.broadcast_tensors(*vals, fref)
        if final_spin is not None:
            final_spin = torch.broadcast_tensors(torch.as_tensor(final_spin, dtype=t.dtype, device=t.device), m1)[0]
        if modes is not None:
            if mode_array is not None:
                raise ValueError("pass only one of modes and mode_array")
            mode_array = modes
        swap = m1 < m2
        m1, m2 = torch.where(swap, m2, m1), torch.where(swap, m1, m2)
        c1, c2 = torch.where(swap, c2, c1), torch.where(swap, c1, c2)
        state = self._tmodel._coeff(m1, m2, c1, c2, final_spin)
        M, eta, delta, S, dchi = state[0], state[1], state[2], state[3], state[4]
        u = (t.reshape((1,) * M.ndim + (-1,)) - tcenter[..., None]) / (M[..., None] * MSUN_SEC)
        sel = _validate_modes(mode_array) if mode_array is not None else self.mode_array
        scale = (M[..., None] * MSUN_SEC) / (dist[..., None] * MPC_SI / C_SI)
        # Match the 22 reference phase convention used by IMRPhenomT.  A
        # zero reference frequency follows the public 20-Hz convention.
        fphase = torch.where(fref > 0, fref, torch.full_like(fref, 20.0))
        tref = self._tmodel.time_at_frequency(fphase, m1, m2, c1, c2, _final_spin_override=final_spin)
        uref = tref / (M * MSUN_SEC)
        phref = self._tmodel._evaluate(uref[..., None], state)[1][..., 0]
        out = {}
        positive = [(l, m) for l, m in sel if m > 0]
        for l, m in positive:
            if m < 0:
                continue
            amp, phase, _ = self._positive(u, state, (l, m), eta, delta, S, dchi, c1, c2, final_spin)
            out[(l, m)] = scale * amp * torch.exp(-1j * (phase - (m / 2) * phref[..., None] + m * pc[..., None]))
        for l, m in sel:
            if m < 0:
                if (l, -m) not in out:
                    amp, phase, _ = self._positive(u, state, (l, -m), eta, delta, S, dchi, c1, c2, final_spin)
                    out[(l, -m)] = scale * amp * torch.exp(-1j * (phase - ((-m) / 2) * phref[..., None] + (-m) * pc[..., None]))
                out[(l, m)] = (-1) ** l * out[(l, -m)].conj()
        return out

    def time_domain(self, times, mass1, mass2, chi1=0., chi2=0., distance=100., inclination=0., phic=0., tc=0., f_ref=20., mode_array=None, final_spin=None, modes=None):
        # Polarizations require the equatorial-symmetry partners.  A public
        # positive-mode selection therefore denotes the corresponding pair.
        if modes is not None:
            if mode_array is not None:
                raise ValueError("pass only one of modes and mode_array")
            mode_array = modes
        selected = None if mode_array is None else tuple(mode_array)
        if selected is not None:
            selected = tuple(dict.fromkeys(selected + tuple((l, -m) for l, m in selected if m > 0)))
        modes_out = self.modes(times, mass1, mass2, chi1, chi2, distance, phic, tc, f_ref=f_ref, mode_array=selected, final_spin=final_spin)
        inc = torch.as_tensor(inclination, dtype=next(iter(modes_out.values())).real.dtype, device=next(iter(modes_out.values())).device)
        # s=-2 harmonics for the five supported l,m; closed forms avoid a scipy dependency.
        c = torch.cos(inc[..., None])
        out = torch.zeros_like(next(iter(modes_out.values())).real), torch.zeros_like(next(iter(modes_out.values())).real)
        # Factorial Wigner-d form of the s=-2 spherical harmonics, evaluated
        # at the author's azimuth pi/2.  Summing signed modes preserves the
        # exact 22 projection while retaining every selected harmonic.
        for (l, m), h in modes_out.items():
            d = torch.zeros_like(c)
            for k in range(0, 20):
                a, b, q, r = l + m - k, k, 2 - m + k, l - 2 - k
                if min(a, b, q, r) < 0:
                    continue
                den = math.factorial(a)*math.factorial(b)*math.factorial(q)*math.factorial(r)
                num = math.factorial(l + 2) * math.factorial(l - 2) * math.factorial(l + m) * math.factorial(l - m)
                coef = math.sqrt(num) / den
                d = d + (-1.)**(k + m - 2) * coef * torch.cos(inc[...,None]/2)**(2*l + m - 2*k - 2) * torch.sin(inc[...,None]/2)**(2 - m + 2*k)
            y = math.sqrt((2*l+1)/(4*math.pi)) * d * ((-1) ** m) * ((1j) ** m)
            out = (out[0] + (h*y).real, out[1] - (h*y).imag)
        return out

    forward = time_domain


__all__ = ["IMRPhenomTHM"]
