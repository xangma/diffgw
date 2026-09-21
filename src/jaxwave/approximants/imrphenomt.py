"""Native JAX implementation of the aligned-spin IMRPhenomT (22) model.

The ansatzes are those of Estelles et al., Phys. Rev. D 103, 124060
(2021), arXiv:2004.08302.  The numerical parameter-space fits are a
native JAX adaptation of the MIT licensed author reference
``imrphenom-dev/phenomxpy`` (commit
4121fafa0484f3db2a7352f4eea697faaaf29f65).  No LALSuite source is used.
"""

from __future__ import annotations

import math
from typing import Union

import jax
import jax.numpy as jnp

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC, PI, EULER_GAMMA


TensorLike = Union[float, jnp.ndarray]


def _t(x, like):
    return x if hasattr(x, "shape") else jnp.asarray(x, dtype=like.dtype)


def _fit(name, *args):
    """Call an attributed author fit, accepting public or underscore spelling."""
    from diffgw_core import phenomt_fits as f

    fn = getattr(f, name, None)
    if fn is None:
        fn = getattr(f, "_" + name, None)
    if fn is None:
        raise AttributeError(f"phenomt_fits has no {name}")
    return fn(*args)


def _fit_any(names, *args):
    last = None
    for name in names:
        try:
            return _fit(name, *args)
        except AttributeError as exc:
            last = exc
    raise last


def _poly_pn_frequency(theta, eta, s1, s2, delta):
    """TaylorT3 dimensionless angular frequency."""
    co = _pn_coefficients(eta, s1, s2, delta)
    t2, t3 = theta**2, theta**3
    t4, t5 = t2**2, t3 * t2
    t6, t7 = t3**2, t4 * t3
    return (
        t3
        / 4
        * (
            1
            + co[..., 0] * t2
            + co[..., 1] * t3
            + co[..., 2] * t4
            + co[..., 3] * t5
            + co[..., 4] * t6
            + 107 / 280 * jnp.log(theta) * t6
            + co[..., 5] * t7
        )
    )


def _pn_phase(theta, eta, delta, s1, s2):
    """TaylorT3 22 phase primitive (the PN part of Eq. A6)."""
    # ``theta`` is already thetabar.  The powers of five enter only in the
    # logarithmic 2.5PN term; multiplying theta by 5**(1/8) here would shift
    # the integration constant and break continuity at the inspiral cut.
    t2, t3 = theta**2, theta**3
    t4, t5 = t2**2, t3 * t2
    t6, t7 = t3**2, t4 * t3
    co = _pn_coefficients(eta, s1, s2, delta)
    logtheta = jnp.log(theta)
    log5 = jnp.log(jnp.asarray(5.0, dtype=theta.dtype))
    return (
        -168
        - 280 * co[..., 0] * t2
        - 420 * co[..., 1] * t3
        - 840 * co[..., 2] * t4
        + 840 * co[..., 3] * (logtheta - log5 / 8) * t5
        - 321 * t6
        + 840 * co[..., 4] * t6
        + 321 * logtheta * t6
        + 420 * co[..., 5] * t7
    ) / (84 * eta * theta**5)


def _pn_coefficients(eta, s1, s2, delta):
    # Coefficients in Appendix A; kept separately for the exact phase primitive.
    e2, e3 = eta * eta, eta * eta * eta
    c1 = 743 / 2688 + 11 * eta / 32
    c15 = -19 * (s1 + s2) * eta / 80 + (-113 * (-s1 * (1 + delta) - s2 * (1 - delta)) - 96 * PI) / 320
    c2 = (
        ((56975 + 61236 * s1 * s1 - 119448 * s1 * s2 + 61236 * s2 * s2) * eta) / 258048
        + 371 * e2 / 2048
        + (1855099 - 3429216 * s1 * s1 * (1 + delta) / 2 - 3429216 * s2 * s2 * (1 - delta) / 2) / 14450688
    )
    c25 = (
        -17 * (s1 + s2) * e2 / 128
        + (-146597 * (-s1 * (1 + delta) - s2 * (1 - delta)) - 46374 * PI) / 129024
        + eta * (-2 * (s1 * (1213 - 63 * delta) + s2 * (1213 + 63 * delta)) + 117 * PI) / 2304
    )
    c3 = (
        -720817631400877 / 288412611379200
        - 16928263 * (s1 * s1 + s2 * s2) / 137625600
        - 16928263 * s1 * s1 * delta / 137625600
        + 16928263 * s2 * s2 * delta / 137625600
        + (-2318475 + 18767224 * s1 * s1 - 54663952 * s1 * s2 + 18767224 * s2 * s2) * e2 / 137625600
        + 235925 * e3 / 1769472
        + 107 * EULER_GAMMA / 280
        - 6127 * PI * (s1 + s2 + s1 * delta - s2 * delta) / 12800
        + eta
        * (
            632550449425
            + 35200873512 * s1 * s1
            - 28527282000 * s1 * s2
            + 9605339856 * s1 * s1 * delta
            - 1512 * s2 * s2 * (-23281001 + 6352738 * delta)
            + 34172264448 * (s1 + s2) * PI
            - 22912243200 * PI * PI
        )
        / 104044953600
        + 53 * PI * PI / 200
        + 107 * math.log(2) / 280
    )
    # The 3.5PN spin terms retain the component-mass factors m1,m2
    # (m1=(1+delta)/2, m2=(1-delta)/2); omitting them changes the fit
    # collocation frequencies appreciably for unequal masses.
    m1, m2 = (1 + delta) / 2, (1 - delta) / 2
    c35 = (
        -12029 * (s1 + s2) * e3 / 92160
        + e2
        * (
            507654 * s1 * s2 * s2
            - 838782 * s2**3
            + s2 * (-840149 + 507654 * s1 * s1 - 870576 * delta)
            + s1 * (-840149 - 838782 * s1 * s1 + 870576 * delta)
            + 1701228 * PI
        )
        / 15482880
        + eta
        * (
            -1134 * s2**3 * (-206917 + 71931 * delta)
            + s1 * (-1496368361 - 429508815 * delta + 1134 * s1 * s1 * (206917 + 71931 * delta))
            - s2 * (1496368361 - 429508815 * delta + 437064012 * s1 * s1 * m1)
            - 437064012 * s1 * s2 * s2 * m2
            - 144 * (488825 + 923076 * s1 * s1 - 1782648 * s1 * s2 + 923076 * s2 * s2) * PI
        )
        / 185794560
        + (
            -2 * s1 * (-6579635551 + 535759434 * s1 * s1) * m1
            + 13159271102 * s2 * m2
            - 1071518868 * s2**3 * m2
            + (-565550067 + 930460608 * s1 * s1 * m1 + 930460608 * s2 * s2 * m2) * PI
        )
        / 1300561920
    )
    return jnp.stack((c1, c15, c2, c25, c3, c35), -1)


def _domega_insp(t, eta, s1, s2, delta, wc):
    t = t if t.ndim > eta.ndim else t.reshape((1,) * eta.ndim + (-1,))
    th = jnp.power(jnp.maximum(-eta[..., None] * t / 5, 1e-30), -1 / 8)
    co = _pn_coefficients(eta[..., None], s1[..., None], s2[..., None], delta[..., None])
    terms = (
        3
        + 5 * co[..., 0] * th**2
        + 6 * co[..., 1] * th**3
        + 7 * co[..., 2] * th**4
        + 8 * co[..., 3] * th**5
        + (107 / 280 + 9 * co[..., 4]) * th**6
        + 10 * co[..., 5] * th**7
    )
    for j in range(6):
        terms = terms + (11 + j) * wc[..., j, None] * th ** (8 + j)
    dwdth = 0.25 * th**2 * terms + 963 / 1120 * th**8 * jnp.log(th)
    dthdt = 0.125 * jnp.power(5 / eta[..., None], 1 / 8) * jnp.power(-t, -9 / 8)
    return dthdt * dwdth


def _phase_insp_raw(t, eta, s1, s2, delta, wc, p5):
    t = t if t.ndim > eta.ndim else t.reshape((1,) * eta.ndim + (-1,))
    tb = jnp.power(jnp.maximum(-eta[..., None] * t, 1e-30), -1 / 8)
    t2, t3 = tb**2, tb**3
    t4, t5 = t2**2, t3 * t2
    t6, t7 = t3**2, t4 * t3
    co = _pn_coefficients(eta[..., None], s1[..., None], s2[..., None], delta[..., None])
    lm = jnp.log(jnp.maximum(-t, 1e-30))
    lt = jnp.log(jnp.asarray(5.0, dtype=t.dtype) ** (1 / 8)) - 0.125 * (jnp.log(eta[..., None]) + lm)
    ps = (
        3 * (-107 + 280 * co[..., 4]) * p5[6]
        + 321 * lt * p5[6]
        + 420 * co[..., 5] * tb * p5[7]
        + 56 * (25 * wc[..., 0, None] + 3 * eta[..., None] * t) * t2
        + 1050 * wc[..., 1, None] * p5[1] * t3
        + 280 * (3 * wc[..., 2, None] + eta[..., None] * co[..., 0] * t) * p5[2] * t4
        + 140 * (5 * wc[..., 3, None] + 3 * eta[..., None] * co[..., 1] * t) * p5[3] * t5
        + 120 * (5 * wc[..., 4, None] + 7 * eta[..., None] * co[..., 2] * t) * p5[4] * t6
        + 525 * wc[..., 5, None] * p5[5] * t7
        + 105 * eta[..., None] * co[..., 3] * t * lm * p5[5] * t7
    )
    return -ps / (84 * p5[5] * eta[..., None] ** 2 * t * tb**7)


class IMRPhenomT:
    """Aligned-spin dominant ``(2,2)`` IMRPhenomT waveform in JAX."""

    @staticmethod
    def _state(m1, m2, chi1, chi2):
        M = m1 + m2
        eta = m1 * m2 / M**2
        delta = (m1 - m2) / M
        # The fit variable is sTotR, normalized by m1^2+m2^2=1-2 eta.
        S = (chi1 * (m1 / M) ** 2 + chi2 * (m2 / M) ** 2) / (1 - 2 * eta)
        dchi = chi1 - chi2
        # The helper provides the published 2017 remnant and Kerr fits.
        af = _fit_any(("IMRPhenomX_FinalSpin2017", "final_spin"), eta, chi1, chi2, delta)
        mf = _fit_any(("IMRPhenomX_FinalMass2017", "final_mass"), eta, chi1, chi2, delta)
        af, mf = _t(af, eta), _t(mf, eta)
        return M, eta, delta, S, dchi, af, mf

    @staticmethod
    def _coeff(m1, m2, chi1, chi2, final_spin=None):
        """Reconstruct the carrier, optionally using a precessing remnant spin."""
        M, eta, delta, S, dchi, af, mf = IMRPhenomT._state(m1, m2, chi1, chi2)
        t0 = _t(_fit_any(("IMRPhenomT_Inspiral_TaylorT3_t0",), eta, S, dchi, delta), eta)
        freq_cp = [_t(_fit_any(("IMRPhenomT_Inspiral_Freq_CP",), eta, S, dchi, delta, i), eta) for i in range(1, 6)]
        peak_amp = _t(_fit_any(("IMRPhenomT_PeakAmp",), eta, S, dchi, delta, 22), eta)
        amp_mid = _t(_fit_any(("IMRPhenomT_Intermediate_Amp_CP1",), eta, S, dchi, delta, 22), eta)
        tshift = jnp.zeros_like(eta)
        c3a = _t(_fit_any(("IMRPhenomT_Ringdown_Amp_C3",), eta, S, dchi, 22), eta)
        omega_peak = _t(_fit_any(("IMRPhenomT_PeakFrequency",), eta, S, dchi, delta, 22), eta)
        c2f = _t(_fit_any(("IMRPhenomT_RD_Freq_C2",), eta, S, dchi, delta, 22), eta)
        c3f = _t(_fit_any(("IMRPhenomT_RD_Freq_C3",), eta, S, dchi, delta, 22), eta)
        mid_freq = _t(_fit_any(("IMRPhenomT_Intermediate_Freq_CP1",), eta, S, dchi, delta, 22), eta)
        # The QNM fits are in units of final mass; the ansatz time is in
        # units of total mass, so only division by mf is required here.
        fr = _t(_fit_any(("IMRPhenomT_fringfit",), af, 22), eta) / mf
        fd = _t(_fit_any(("IMRPhenomT_fdampfit",), af, 22), eta) / mf
        fd2 = _t(_fit_any(("IMRPhenomT_fdampn2fit",), af, 22), eta) / mf
        # The NNLO precessing remnant changes only ringdown (TPHM Sec. II C).
        # Preserve the calibrated aligned-spin inspiral and merger joins.
        rd_fr, rd_fd, rd_fd2 = fr, fd, fd2
        if final_spin is not None:
            rd_af = jnp.asarray(final_spin, dtype=eta.dtype) + jnp.zeros_like(eta)
            rd_fr = _t(_fit_any(("IMRPhenomT_fringfit",), rd_af, 22), eta) / mf
            rd_fd = _t(_fit_any(("IMRPhenomT_fdampfit",), rd_af, 22), eta) / mf
            rd_fd2 = _t(_fit_any(("IMRPhenomT_fdampn2fit",), rd_af, 22), eta) / mf

        th = jnp.asarray((0.33, 0.45, 0.55, 0.65, 0.75, 0.82), dtype=eta.dtype)
        qmat = jnp.stack([th ** (8 + j) for j in range(6)], -1)
        tearly = -5.0 / (eta * th[0] ** 8)
        theta0 = jnp.power(jnp.maximum(eta * (t0 - tearly) / 5.0, 1.0e-30), -1 / 8)
        vals = jnp.stack([_poly_pn_frequency(theta0, eta, chi1, chi2, delta)] + freq_cp, -1)
        pvals = _poly_pn_frequency(th, eta[..., None], chi1[..., None], chi2[..., None], delta[..., None])
        rhs = 4.0 / th**3 * (vals - pvals)
        wc = jnp.linalg.solve(jnp.broadcast_to(qmat, rhs.shape[:-1] + qmat.shape), rhs[..., None])[..., 0]

        # Merger frequency coefficients, Eqs. (28-31) of the reference.
        tcut = -26.982976386771437 / eta
        tcp = -5.0 / (eta * jnp.asarray(0.95, dtype=eta.dtype) ** 8)
        wcut = IMRPhenomT._omega_insp(tcut[..., None], eta, delta, chi1, chi2, wc)[..., 0]
        dwcut = _domega_insp(tcut[..., None], eta, chi1, chi2, delta, wc)[..., 0]
        wr = 2 * PI * fr
        alpha = 2 * PI * fd
        c1f = (1 + c3f) * (wr - omega_peak) / (c2f * c3f)
        dwpeak = -c1f * c2f**2 * c3f / (1 + c3f) ** 2 / wr
        acut = jnp.arcsinh(alpha * tcut)
        amer = jnp.arcsinh(alpha * tcp)
        dencut = jnp.sqrt(1 + (alpha * tcut) ** 2)
        b0 = 1 - (wcut / wr) - (1 - omega_peak / wr) - dwpeak / alpha * acut
        b1 = 1 - mid_freq / wr - (1 - omega_peak / wr) - dwpeak / alpha * amer
        b2 = -dwcut / wr - dwpeak / dencut
        amat = jnp.stack(
            (
                jnp.stack((acut**2, acut**3, acut**4), -1),
                jnp.stack((amer**2, amer**3, amer**4), -1),
                jnp.stack((2 * alpha * acut / dencut, 3 * alpha * acut**2 / dencut, 4 * alpha * acut**3 / dencut), -1),
            ),
            -2,
        )
        mc = jnp.linalg.solve(amat, jnp.stack((b0, b1, b2), -1)[..., None])[..., 0]

        # Amplitude reconstruction is kept in the MIT-derived private helper;
        # it consumes analytic piecewise omega and domega callbacks.
        from ._phenomt_amplitude import amplitude_coefficients

        def omega_cb(tt):
            ti = jnp.minimum(tt, tcut[..., None] - 1.0e-10)
            wi = IMRPhenomT._omega_insp(ti, eta, delta, chi1, chi2, wc)
            tm = jnp.clip(tt, tcut[..., None], 0.0)
            x = jnp.arcsinh(alpha[..., None] * tm)
            wm = wr[..., None] * (
                1
                - (
                    1
                    - omega_peak[..., None] / wr[..., None]
                    + x
                    * (
                        dwpeak[..., None] / alpha[..., None]
                        + x * (mc[..., 0, None] + x * (mc[..., 1, None] + x * mc[..., 2, None]))
                    )
                )
            )
            tr = jnp.maximum(tt, jnp.zeros_like(tt))
            er = jnp.exp(-c2f[..., None] * tr)
            wd = wr[..., None] - c1f[..., None] * c2f[..., None] * c3f[..., None] * er / (1 + c3f[..., None] * er)
            return jnp.where(tt < tcut[..., None], wi, jnp.where(tt < 0, wm, wd))

        def domega_cb(tt):
            ti = jnp.minimum(tt, tcut[..., None] - 1.0e-10)
            di = _domega_insp(ti, eta, chi1, chi2, delta, wc)
            tm = jnp.clip(tt, tcut[..., None], 0.0)
            x = jnp.arcsinh(alpha[..., None] * tm)
            dx = alpha[..., None] / jnp.sqrt(1 + (alpha[..., None] * tm) ** 2)
            dqx = (
                dwpeak[..., None] / alpha[..., None]
                + 2 * mc[..., 0, None] * x
                + 3 * mc[..., 1, None] * x**2
                + 4 * mc[..., 2, None] * x**3
            )
            dm = -wr[..., None] * dqx * dx
            tr = jnp.maximum(tt, jnp.zeros_like(tt))
            er = jnp.exp(-c2f[..., None] * tr)
            dd = c1f[..., None] * c2f[..., None] ** 2 * c3f[..., None] * er / (1 + c3f[..., None] * er) ** 2
            return jnp.where(tt < tcut[..., None], di, jnp.where(tt < 0, dm, dd))

        amp_state = amplitude_coefficients(eta, delta, chi1, chi2, fd, fd2, omega_cb, domega_cb, rd_fd, rd_fd2)

        # Analytic phase offsets make all three regions C0 continuous.
        p5 = jnp.asarray(5.0, dtype=eta.dtype) ** (jnp.arange(8, dtype=eta.dtype) / 8)
        theta_i = jnp.power(jnp.maximum(eta * (t0 - tearly), 1.0e-30), -1 / 8)
        phoff_i = (
            _pn_phase(theta_i, eta, delta, chi1, chi2)
            - _phase_insp_raw(tearly[..., None], eta, chi1, chi2, delta, wc, p5)[..., 0]
        )
        phoff_m = (
            phoff_i
            + _phase_insp_raw(tcut[..., None], eta, chi1, chi2, delta, wc, p5)[..., 0]
            - IMRPhenomT._phase_inter_raw(tcut, alpha, wr, omega_peak, dwpeak, mc)
        )
        phoff_r = IMRPhenomT._phase_inter_raw(jnp.zeros_like(eta), alpha, wr, omega_peak, dwpeak, mc) + phoff_m
        return (
            M,
            eta,
            delta,
            S,
            dchi,
            chi1,
            chi2,
            wc,
            amp_state,
            peak_amp,
            amp_mid,
            tshift,
            c3a,
            omega_peak,
            c2f,
            c3f,
            mid_freq,
            fr,
            fd,
            fd2,
            alpha,
            wr,
            c1f,
            dwpeak,
            mc,
            tcut,
            phoff_i,
            phoff_m,
            phoff_r,
            p5,
            2 * PI * rd_fr,
            (1 + c3f) * (2 * PI * rd_fr - omega_peak) / (c2f * c3f),
        )

    @staticmethod
    def _omega_insp(t, eta, delta, s1, s2, wc):
        if t.ndim <= eta.ndim:
            t = t.reshape((1,) * eta.ndim + (-1,))
        th = jnp.power(jnp.maximum(-eta[..., None] * t / 5, 1e-30), -1 / 8)
        return _poly_pn_frequency(th, eta[..., None], s1[..., None], s2[..., None], delta[..., None]) + th**3 / 4 * sum(
            wc[..., j, None] * th ** (8 + j) for j in range(6)
        )

    @staticmethod
    def _phase_inter_raw(t, alpha, wr, op, dwp, mc):
        x = jnp.arcsinh(alpha * t)
        x2, x3, x4 = x * x, x * x * x, x * x * x * x
        root = jnp.sqrt(1 + (alpha * t) ** 2)
        return wr * t * (
            1
            - (
                2 * mc[..., 0]
                + 24 * mc[..., 2]
                + (6 * mc[..., 1] + dwp / alpha) * x
                + (1 - op / wr)
                + (mc[..., 0] + 12 * mc[..., 2]) * x2
                + mc[..., 1] * x3
                + mc[..., 2] * x4
            )
        ) - wr / alpha * root * (
            -dwp / alpha
            - 6 * mc[..., 1]
            - x * (2 * mc[..., 0] + 24 * mc[..., 2])
            - 3 * mc[..., 1] * x2
            - 4 * mc[..., 2] * x3
        )

    @staticmethod
    def _evaluate(u, state):
        (
            M,
            eta,
            delta,
            S,
            dchi,
            s1,
            s2,
            wc,
            amp_state,
            peak,
            amid,
            tshift,
            c3a,
            op,
            c2f,
            c3f,
            midf,
            fr,
            fd,
            fd2,
            alpha,
            wr,
            c1f,
            dwp,
            mc,
            tcut,
            phoi,
            phom,
            phor,
            p5,
            rd_wr,
            rd_c1f,
        ) = state
        ui = jnp.minimum(u, tcut[..., None])
        win = IMRPhenomT._omega_insp(ui, eta, delta, s1, s2, wc)
        um = jnp.clip(u, tcut[..., None], 0.0)
        ur = jnp.maximum(u, jnp.zeros_like(u))
        ex = jnp.exp(-c2f[..., None] * ur)
        wring = rd_wr[..., None] - rd_c1f[..., None] * c2f[..., None] * c3f[..., None] * ex / (1 + c3f[..., None] * ex)
        x = jnp.arcsinh(alpha[..., None] * um)
        wm = wr[..., None] * (
            1
            - (
                1
                - op[..., None] / wr[..., None]
                + x
                * (
                    dwp[..., None] / alpha[..., None]
                    + x * (mc[..., 0, None] + x * (mc[..., 1, None] + x * mc[..., 2, None]))
                )
            )
        )
        omega = jnp.where(u < tcut[..., None], win, jnp.where(u < 0, wm, wring))
        pi = _phase_insp_raw(ui, eta, s1, s2, delta, wc, p5) + phoi[..., None]
        pm = (
            IMRPhenomT._phase_inter_raw(
                um, alpha[..., None], wr[..., None], op[..., None], dwp[..., None], mc[..., None, :]
            )
            + phom[..., None]
        )
        er = jnp.exp(-c2f[..., None] * ur)
        pr = (
            rd_c1f[..., None] * jnp.log((1 + c3f[..., None] * er) / (1 + c3f[..., None]))
            + rd_wr[..., None] * ur
            + phor[..., None]
        )
        phase = jnp.where(u < tcut[..., None], pi, jnp.where(u < 0, pm, pr))
        return omega, phase, amp_state

    @staticmethod
    def _domega_full(u, state):
        """Analytic derivative of the piecewise IMR angular frequency."""
        (
            M,
            eta,
            delta,
            S,
            dchi,
            s1,
            s2,
            wc,
            amp_state,
            peak,
            amid,
            tshift,
            c3a,
            op,
            c2f,
            c3f,
            midf,
            fr,
            fd,
            fd2,
            alpha,
            wr,
            c1f,
            dwp,
            mc,
            tcut,
            phoi,
            phom,
            phor,
            p5,
            rd_wr,
            rd_c1f,
        ) = state
        ui = jnp.minimum(u, tcut[..., None])
        um = jnp.clip(u, tcut[..., None], 0.0)
        ur = jnp.maximum(u, jnp.zeros_like(u))
        di = _domega_insp(ui, eta, s1, s2, delta, wc)
        x = jnp.arcsinh(alpha[..., None] * um)
        dx = alpha[..., None] / jnp.sqrt(1 + (alpha[..., None] * um) ** 2)
        dqx = (
            dwp[..., None] / alpha[..., None]
            + 2 * mc[..., 0, None] * x
            + 3 * mc[..., 1, None] * x**2
            + 4 * mc[..., 2, None] * x**3
        )
        dm = -wr[..., None] * dqx * dx
        er = jnp.exp(-c2f[..., None] * ur)
        dd = rd_c1f[..., None] * c2f[..., None] ** 2 * c3f[..., None] * er / (1 + c3f[..., None] * er) ** 2
        return jnp.where(u < tcut[..., None], di, jnp.where(u < 0, dm, dd))

    def time_domain(
        self,
        times: TensorLike,
        mass1: TensorLike,
        mass2: TensorLike,
        chi1: TensorLike = 0.0,
        chi2: TensorLike = 0.0,
        distance: TensorLike = 100.0,
        inclination: TensorLike = 0.0,
        phic: TensorLike = 0.0,
        tc: TensorLike = 0.0,
        f_ref: TensorLike = 20.0,
        *,
        _final_spin_override=None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return h_plus, h_cross on times in seconds relative to the peak.

        Masses are solar masses, distance is Mpc, and f_ref is Hz. Parameter
        batches precede the shared time axis. This method applies no start mask.
        """
        t = jnp.asarray(times, dtype=jnp.float64)
        m1, m2, c1, c2, d, inc, pc, tcenter, fr = [
            jnp.asarray(v, dtype=t.dtype) for v in (mass1, mass2, chi1, chi2, distance, inclination, phic, tc, f_ref)
        ]
        m1, m2, c1, c2, d, inc, pc, tcenter, fr = jnp.broadcast_arrays(m1, m2, c1, c2, d, inc, pc, tcenter, fr)
        state = self._coeff(m1, m2, c1, c2, final_spin=_final_spin_override)
        M = state[0]
        u = (t.reshape((1,) * M.ndim + (-1,)) - tcenter[..., None]) / (M[..., None] * MSUN_SEC)
        omega, phase, amp_state = self._evaluate(u, state)
        from ._phenomt_amplitude import amplitude

        amp = amplitude(u, amp_state, lambda tt: self._evaluate(tt, state)[0])
        # f_ref=0 follows the public convention and means the default 20 Hz
        # lower/reference frequency.  Subtract the analytic phase at that
        # frequency, as the author implementation does.
        fphase = jnp.where(fr > 0, fr, jnp.full_like(fr, 20.0))
        tref = self.time_at_frequency(fphase, m1, m2, c1, c2, _final_spin_override=_final_spin_override)
        uref = tref / (M * MSUN_SEC)
        phref = self._evaluate(uref[..., None], state)[1][..., 0]
        ph = phase - phref[..., None] + 2.0 * pc[..., None]
        scale = (M[..., None] * MSUN_SEC) / (d[..., None] * MPC_SI / C_SI)
        y22 = math.sqrt(5.0 / (4.0 * math.pi))
        ci = jnp.cos(inc[..., None])
        pref = -scale * y22 * amp
        return pref * (1 + ci**2) * 0.5 * jnp.cos(ph), pref * ci * jnp.sin(ph)

    forward = time_domain
    __call__ = time_domain

    def frequency(self, times, mass1, mass2, chi1=0.0, chi2=0.0, *, _final_spin_override=None):
        """Instantaneous gravitational-wave frequency in Hz at peak-relative times."""
        t = jnp.asarray(times, dtype=jnp.float64)
        m1 = jnp.asarray(mass1, dtype=t.dtype)
        m2 = jnp.asarray(mass2, dtype=t.dtype)
        c1 = jnp.asarray(chi1, dtype=t.dtype)
        c2 = jnp.asarray(chi2, dtype=t.dtype)
        m1, m2, c1, c2 = jnp.broadcast_arrays(m1, m2, c1, c2)
        M = m1 + m2
        u = t.reshape((1,) * M.ndim + (-1,)) / (M[..., None] * MSUN_SEC)
        return self._evaluate(u, self._coeff(m1, m2, c1, c2, final_spin=_final_spin_override))[0] / (2 * PI * M[..., None] * MSUN_SEC)

    def time_at_frequency(self, frequency, mass1, mass2, chi1=0.0, chi2=0.0, *, _final_spin_override=None):
        """Return peak-relative time at a requested instantaneous frequency.

        The bracket is grown from the Newtonian inspiral estimate so that it
        remains useful for low-frequency BNS requests.  Bisection itself is
        stopped from the differentiation graph; an implicit Newton correction
        then restores the derivative through the frequency equation while
        keeping JIT and batched execution fully functional.
        """
        f = jnp.asarray(frequency, dtype=jnp.float64)
        m1 = jnp.asarray(mass1, dtype=f.dtype)
        m2 = jnp.asarray(mass2, dtype=f.dtype)
        c1 = jnp.asarray(chi1, dtype=f.dtype)
        c2 = jnp.asarray(chi2, dtype=f.dtype)
        m1, m2, c1, c2, f = jnp.broadcast_arrays(m1, m2, c1, c2, f)
        M = m1 + m2
        target = 2 * PI * f * M * MSUN_SEC
        st = self._coeff(m1, m2, c1, c2, final_spin=_final_spin_override)

        # Eager calls give a useful exception.  Under jit these conditions are
        # represented by ``valid`` below, yielding NaN rather than selecting a
        # clipped endpoint when the requested root does not exist.
        concrete = not any(isinstance(v, jax.core.Tracer) for v in (f, m1, m2, c1, c2, _final_spin_override))
        valid = jnp.isfinite(f) & (f > 0) & (target < st[30])
        if concrete:
            if not bool(jnp.all(jnp.isfinite(f) & (f > 0))):
                raise ValueError("frequency must be finite and positive")
            if bool(jnp.any(target >= st[30])):
                raise ValueError("frequency is above the supported IMRPhenomT range")

        # Newtonian leading-order time, buffered as in the author solver.
        lo0 = -5.0 / (256.0 * st[1] * jnp.power(target / 2.0, 8.0 / 3.0)) * 1.2
        hi0 = jnp.zeros_like(target)

        def grow_hi(_, hi):
            om_hi = self._evaluate(hi[..., None], st)[0][..., 0]
            need = om_hi < target
            return jnp.where(need, jnp.where(hi == 0, jnp.ones_like(hi), 2 * hi), hi)

        def grow_lo(_, lo):
            om_lo = self._evaluate(lo[..., None], st)[0][..., 0]
            return jnp.where(om_lo > target, lo * 2.0, lo)

        hi = jax.lax.fori_loop(0, 16, grow_hi, hi0)
        lo = jax.lax.fori_loop(0, 12, grow_lo, lo0)

        def bisect(_, bounds):
            lo_i, hi_i = bounds
            mid = (lo_i + hi_i) / 2
            om = self._evaluate(mid[..., None], st)[0][..., 0]
            below = om < target
            return jnp.where(below, mid, lo_i), jnp.where(below, hi_i, mid)

        lo, hi = jax.lax.fori_loop(0, 80, bisect, (lo, hi))
        root = jax.lax.stop_gradient((lo + hi) / 2)
        om = self._evaluate(root[..., None], st)[0][..., 0]
        dom = self._domega_full(root[..., None], st)[..., 0]
        root = root + (target - om) / jnp.maximum(dom, jnp.finfo(root.dtype).tiny)
        root = jnp.where(valid, root, jnp.nan)
        return root * M * MSUN_SEC


__all__ = ["IMRPhenomT"]
