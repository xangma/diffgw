"""Clean-room phase diagnostic for the public IMRPhenomHM black box.

The predicted phases are London's Eqs. (5), (8), (9), (13), (14), with the
Khan Sec. V C1 attachment.  This file deliberately uses only public callable
APIs for the native comparison.
"""
import hashlib
import inspect
import json
import pathlib
import platform

import torch
import lal
import lalsimulation

from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenom_hm_phase_clean import (
    mapped_phase_c1, phenomd_eq14_phase_delta_c1,
    phenomd_eq14_phase_delta_derivative, low_frequency_offset,
)
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm, SUPPORTED_MODES
from torchwave.approximants.kerr_qnm_london_published import dimensionless_qnm as london_qnm

MTSUN_SEC = 4.925490947641267e-6
CASES = ((40., 15., .3, -.2, 20.), (55., 11., -.4, .2, 27.),
         (32., 18., -.6, .45, 25.), (70., 8., .1, -.5, 30.))
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrapped(a):
    return torch.angle(torch.exp(1j * a))


def _binary_modes(freqs, m1, m2, s1, s2, fref, phic):
    """Read public LAL mode nodes; no implementation internals are used."""
    fv = lal.CreateREAL8Vector(len(freqs))
    fv.data[:] = freqs.detach().cpu().numpy()
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.
    head = lalsimulation.SimIMRPhenomHMGethlmModes(
        fv, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0., 0., s1,
        0., 0., s2, float(phic), df, float(fref), lal.CreateDict())
    out = {}
    n = head
    while n:
        out[(int(n.l), int(n.m))] = torch.tensor(n.mode.data.data, dtype=torch.complex128)
        n = n.next
    return out


def _component_qnm(which, final_mass_msun, final_spin, ell, m):
    """Public EOB QNM output converted from angular /s to dimensionless Mf*f."""
    vec = lal.CreateCOMPLEX16Vector(1)
    fn = getattr(lalsimulation, which)
    try:
        rc = fn(vec, float(final_mass_msun), float(final_spin), ell, m, 1)
    except (RuntimeError, ValueError):
        return None
    if rc != 0:
        return None
    assert rc == 0 and torch.isfinite(torch.tensor(vec.data[0].real))
    scale = float(final_mass_msun) * MTSUN_SEC
    # The component returns angular /s; divide by 2*pi and multiply by final M.
    return float(vec.data[0].real) * scale / (2 * torch.pi), float(vec.data[0].imag) * scale / (2 * torch.pi)


def main():
    d, hm = IMRPhenomD(), IMRPhenomHM()
    c1 = _component_qnm("SimIMREOBGenerateQNMFreqV2fromFinal", 1., 0., 2, 2)
    c2 = _component_qnm("SimIMREOBGenerateQNMFreqV2fromFinal", 2., 0., 2, 2)
    assert abs(c1[0] / c2[0] - 1.) < 1e-12
    kw0, kd0 = dimensionless_qnm(2, 2, 0, torch.tensor(0., dtype=torch.float64))
    assert abs(c1[0] - float(kw0) / (2 * torch.pi)) < 1e-6
    # Manufactured nonlinear-carrier check: an affine carrier gauge A+B*x
    # propagates through the C1 map as (m/2)A+B*x on the low branch.
    gx = torch.linspace(.005, .015, 9, dtype=torch.float64)
    ga, gb, gm = .37, -.41, 3
    gc = lambda y: y*y
    gs = mapped_phase_c1(gx, .02, .07, .06, gm, gc, ell=gm)
    gshift = mapped_phase_c1(gx, .02, .07, .06, gm,
                             lambda y: gc(y) + ga + gb*y, ell=gm) - gs
    torch.testing.assert_close(gshift, (gm / 2) * ga + gb * gx,
                               atol=1e-12, rtol=1e-12)
    f_hz = torch.linspace(20., 512., 1025, dtype=torch.float64)
    rows = []
    checks = []
    for m1, m2, s1, s2, fref in CASES:
        total = m1 + m2
        msec = total * MTSUN_SEC
        eta = m1 * m2 / total ** 2
        chi_eff = (m1 * s1 + m2 * s2) / total
        chi_pn = chi_eff - 38. * eta / 113. * (s1 + s2)
        args = tuple(torch.tensor(v, dtype=torch.float64) for v in (m1, m2, s1, s2))
        slope_peak = float(d.compute_slope_peak(*args))
        rem = d.remnant_properties(*args)
        mf, af, fr22, fd22 = (float(x) for x in rem)
        fr22, fd22 = float(fr22), float(fd22)
        hm_fr22, hm_fd22 = (float(x) for x in hm.mode_qnm_frequencies(2, 2, *args))
        # Independent Kerr eigenvalues are Re/Im(Mf*omega), converted to the
        # native total-Mf convention using the same D remnant mass.
        kerr_ratio = total / mf
        kerr_qnm = {}
        london_qnm_modes = {}
        for kl, km in MODES:
            if (kl, km, 0) in SUPPORTED_MODES:
                kw, kd = dimensionless_qnm(kl, km, 0, torch.tensor(af, dtype=torch.float64))
                kerr_qnm[(kl, km)] = (float(kw) / (2 * torch.pi) * kerr_ratio,
                                      float(kd) / (2 * torch.pi) * kerr_ratio)
            try:
                lw = london_qnm(af, kl, km)
                london_qnm_modes[(kl, km)] = (lw.real / (2 * torch.pi) * kerr_ratio,
                                              lw.imag / (2 * torch.pi) * kerr_ratio)
            except ValueError:
                pass
        component_qnm = {name: {mode: _component_qnm(name, mf, af, *mode)
                                for mode in MODES}
                         for name in ("SimIMREOBGenerateQNMFreqV2fromFinal",
                                      "SimIMREOBGenerateQNMFreqV5fromFinal")}
        # HM inputs use total-M*f, so convert the component Mf*f values by M/Mf.
        component_qnm = {name: {mode: None if value is None else
                                 (value[0] * kerr_ratio, value[1] * kerr_ratio)
                                 for mode, value in values.items()}
                         for name, values in component_qnm.items()}
        # D is called in Hz; x is the dimensionless M*f coordinate.
        def carrier0(x):
            return d(x / msec, m1, m2, s1, s2, phic=0., tc=0., f_ref=fref,
                     return_phase=True)[2]
        def carrier_phys(x):
            return -carrier0(x)
        p_direct = d(f_hz, m1, m2, s1, s2, phic=0., tc=0., f_ref=fref,
                     return_phase=True)[2]
        p_scaled = carrier0(f_hz * msec)
        torch.testing.assert_close(p_scaled, p_direct, atol=1e-10, rtol=1e-12)
        hp = d(f_hz, m1, m2, s1, s2, phic=0., tc=0., f_ref=fref,
               return_phase=True)[0]
        torch.testing.assert_close(hp, torch.abs(hp) * torch.exp(-1j * p_direct),
                                   atol=1e-30, rtol=2e-12)
        checks.append("carrier Hz/Msec identity and D hp exp(-i*p0)")
        native = hm(f_hz, mass1=m1, mass2=m2, chi1=s1, chi2=s2,
                    distance=500., inclination=.8, phic=0., tc=0., f_ref=fref,
                    f_lower=20., return_unprojected_modes=True)[2]
        a4 = float(d.eval_table_v("alpha4", torch.tensor(eta, dtype=torch.float64), torch.tensor(chi_pn, dtype=torch.float64)))
        a5 = float(d.eval_table_v("alpha5", torch.tensor(eta, dtype=torch.float64), torch.tensor(chi_pn, dtype=torch.float64)))
        binary = None
        binary_error = None
        try:
            fv = lal.CreateREAL8Vector(len(f_hz))
            fv.data[:] = f_hz.numpy()
            b = lalsimulation.SimIMRPhenomHMGethlmModes(
                fv, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0., 0., s1,
                0., 0., s2, 0., float(f_hz[1] - f_hz[0]), fref,
                lal.CreateDict())
            binary = {}
            n = b
            while n:
                binary[(int(n.l), int(n.m))] = torch.tensor(n.mode.data.data, dtype=torch.complex128)
                n = n.next
            for _mode, _zmode in binary.items():
                assert torch.isfinite(_zmode.real).all() and torch.isfinite(_zmode.imag).all()
                assert torch.all(torch.abs(_zmode) > 0), f"zero/padded binary mode {_mode}"
            n22 = native[(2, 2)] / torch.abs(native[(2, 2)])
            b22 = binary[(2, 2)] / torch.abs(binary[(2, 2)])
            binary22_l2 = float(torch.linalg.vector_norm(n22 - b22) / len(n22) ** .5)
            if binary22_l2 >= 1.e-3:
                raise AssertionError(f"binary/native 22 convention mismatch: {binary22_l2}")
        except Exception as exc:
            binary = None
            binary_error = {"type": type(exc).__name__, "message": str(exc),
                            "callable": "SimIMRPhenomHMGethlmModes"}
        mode_rows = {}
        for ell, m in MODES:
            if (ell, m) == (2, 2):
                frlm, fdlm = fr22, fd22
            else:
                frlm, fdlm = (float(x) for x in hm.mode_qnm_frequencies(ell, m, *args))
            f0 = .014 * frlm / fr22
            x = f_hz * msec
            base = mapped_phase_c1(x, f0, frlm, fr22, m, carrier_phys, ell=ell)
            carrier_lm = lambda y: carrier_phys(y) - phenomd_eq14_phase_delta_c1(
                y, eta, chi_pn, fr22, frlm, fd22, fdlm, a4, a5)
            eq14 = mapped_phase_c1(x, f0, frlm, fr22, m, carrier_lm, ell=ell)
            positive_lm = lambda y: carrier0(y) + phenomd_eq14_phase_delta_c1(
                y, eta, chi_pn, fr22, frlm, fd22, fdlm, a4, a5)
            corrected = -mapped_phase_c1(x, f0, frlm, fr22, m, positive_lm, ell=ell)
            # AMPLITUDE's published transition is .018; retain it as a
            # diagnostic control to expose the branch-choice effect.
            amp_join_control = mapped_phase_c1(x, .018 * frlm / fr22, frlm, fr22,
                                               m, carrier_phys, ell=ell)
            if (ell, m) == (2, 2):
                torch.testing.assert_close(phenomd_eq14_phase_delta_c1(
                    x, eta, chi_pn, fr22, fr22, fd22, fd22, a4, a5),
                    torch.zeros_like(x), atol=0, rtol=0)
                # With ell=m=2 and identical QNM endpoints, the map is the
                # identity on every branch, independently of the join value.
                torch.testing.assert_close(
                    torch.exp(1j * mapped_phase_c1(x, f0, fr22, fr22, 2, carrier0, ell=2)),
                    torch.exp(1j * carrier0(x)), atol=1e-11, rtol=1e-12)
                torch.testing.assert_close(
                    torch.exp(1j * mapped_phase_c1(x, .014, hm_fr22, hm_fr22, 2, carrier0, ell=2)),
                    torch.exp(1j * carrier0(x)), atol=1e-11, rtol=1e-12)
                checks.append("22 D-QNM and HM-QNM map identity")
                z22 = native[(2, 2)] / torch.abs(native[(2, 2)])
                e = torch.exp(1j * base)
                direct = torch.linalg.vector_norm(z22 - e) / len(e) ** .5
                conjugate = torch.linalg.vector_norm(z22 - torch.conj(e)) / len(e) ** .5
                checks.append("native 22 direct-vs-conjugate convention recorded")
                convention = {"direct_l2": float(direct), "conjugate_l2": float(conjugate),
                               "selected": "direct" if direct <= conjugate else "conjugate"}
            else:
                convention = None
            z = native[(ell, m)]
            bmode = None if binary is None else binary[(ell, m)]
            native_valid = torch.abs(z) > torch.max(torch.abs(z)) * 1e-12
            valid = native_valid if bmode is None else (native_valid & (torch.abs(bmode) > torch.max(torch.abs(bmode)) * 1e-12))
            def empty_or(fn, mask):
                return {"sample_count": int(mask.sum())} if not bool(mask.any()) else dict(fn(mask), sample_count=int(mask.sum()))
            def metrics(pred, mask=valid):
                if not bool(mask.any()): return {"sample_count": 0}
                zn = z[mask] / torch.abs(z[mask])
                pn = torch.exp(1j * pred[mask])
                return {"raw_wrapped_phase_rms_rad": float(torch.sqrt(torch.mean(_wrapped(torch.angle(z[mask]) - pred[mask]) ** 2))),
                        "relative_complex_l2": float(torch.linalg.vector_norm(zn - pn) / len(pn) ** .5), "sample_count": int(mask.sum())}
            def compare(zother, mask=valid):
                if not bool(mask.any()): return {"sample_count": 0}
                za = z[mask] / torch.abs(z[mask])
                zb = zother[mask] / torch.abs(zother[mask])
                return {"raw_wrapped_phase_rms_rad": float(torch.sqrt(torch.mean(_wrapped(torch.angle(z[mask]) - torch.angle(zother[mask])) ** 2))),
                        "relative_complex_l2": float(torch.linalg.vector_norm(za - zb) / len(za) ** .5), "sample_count": int(mask.sum())}
            def clean_metrics(pred, mask=valid):
                if not bool(mask.any()): return {"sample_count": 0}
                pn = torch.exp(1j * pred[mask])
                zn = z[mask] / torch.abs(z[mask])
                return {"raw_wrapped_phase_rms_rad": float(torch.sqrt(torch.mean(_wrapped(torch.angle(z[mask]) - pred[mask]) ** 2))),
                        "relative_complex_l2": float(torch.linalg.vector_norm(zn - pn) / len(pn) ** .5), "sample_count": int(mask.sum())}
            def binary_clean(pred, mask=valid):
                if binary is None or not bool(mask.any()): return {"sample_count": 0}
                zb = bmode[mask] / torch.abs(bmode[mask])
                pn = torch.exp(1j * pred[mask])
                return {"raw_wrapped_phase_rms_rad": float(torch.sqrt(torch.mean(_wrapped(torch.angle(bmode[mask]) - pred[mask]) ** 2))),
                        "relative_complex_l2": float(torch.linalg.vector_norm(zb - pn) / len(pn) ** .5), "sample_count": int(mask.sum())}
            def raw_metrics(zother, pred, mask):
                if not bool(mask.any()): return {"sample_count": 0}
                zn = zother[mask] / torch.abs(zother[mask]); pn = torch.exp(1j * pred[mask])
                return {"raw_wrapped_phase_rms_rad": float(torch.sqrt(torch.mean(_wrapped(torch.angle(zother[mask]) - pred[mask]) ** 2))),
                        "relative_complex_l2": float(torch.linalg.vector_norm(zn-pn)/len(pn)**.5), "sample_count": int(mask.sum())}
            low = valid & (x < f0)
            mid = valid & (x >= f0) & (x <= frlm)
            high = valid & (x > frlm)
            gauge0 = (m / 2. - 1.) * slope_peak * msec * fref
            corrected_with_gauge = corrected + gauge0
            f018 = .018 * frlm / fr22
            corrected018 = -mapped_phase_c1(x, f018, frlm, fr22, m, positive_lm, ell=ell) + gauge0
            # Published .014/.018 transition controls using the independently
            # supplied HM (Berti) 22 QNM as map denominator.  The D carrier
            # and its Eq14 QNM parameters remain unchanged in these controls.
            map_frlm_hm = hm_fr22 if (ell, m) == (2, 2) else frlm
            f014_hm22 = .014 * map_frlm_hm / hm_fr22
            f018_hm22 = .018 * map_frlm_hm / hm_fr22
            corrected014_hm22 = -mapped_phase_c1(
                x, f014_hm22, map_frlm_hm, hm_fr22, m, positive_lm, ell=ell) + gauge0
            corrected018_hm22 = -mapped_phase_c1(
                x, f018_hm22, map_frlm_hm, hm_fr22, m, positive_lm, ell=ell) + gauge0
            hm_ratio = hm_fr22 / map_frlm_hm
            positive_lm_hm_ratio = lambda y: carrier0(y) + phenomd_eq14_phase_delta_c1(
                y, eta, chi_pn, fr22, frlm, fd22, fdlm, a4, a5, qnm_ratio=hm_ratio)
            corrected018_hm22_ratio = -mapped_phase_c1(
                x, f018_hm22, map_frlm_hm, hm_fr22, m, positive_lm_hm_ratio, ell=ell) + gauge0
            kerr_available = (ell, m) in kerr_qnm
            if kerr_available:
                kfr22, kfd22 = kerr_qnm[(2, 2)]
                kfrlm, kfdlm = kerr_qnm[(ell, m)]
                kmap_f0 = .018 * kfrlm / kfr22
                positive_lm_kerr = lambda y: carrier0(y) + phenomd_eq14_phase_delta_c1(
                    y, eta, chi_pn, fr22, kfrlm, fd22, kfdlm, a4, a5,
                    qnm_ratio=kfr22 / kfrlm)
                corrected018_kerr = -mapped_phase_c1(
                    x, kmap_f0, kfrlm, kfr22, m, positive_lm_kerr, ell=ell) + gauge0
                k_low = valid & (x < kmap_f0)
                k_mid = valid & (x >= kmap_f0) & (x <= kfrlm)
                k_high = valid & (x > kfrlm)
                kerr_probe_rows = []
                if (ell, m) in ((2, 1), (3, 3)):
                    for xp in (.02, .04, .06, .08, .12):
                        xt = torch.tensor([xp], dtype=torch.float64)
                        zbk = _binary_modes(xt / msec, m1, m2, s1, s2, fref, 0.)[(ell, m)][0]
                        assert torch.isfinite(zbk.real) and torch.isfinite(zbk.imag) and torch.abs(zbk) > 0
                        predk = -mapped_phase_c1(xt, kmap_f0, kfrlm, kfr22, m,
                                                 positive_lm_kerr, ell=ell)[0] + gauge0
                        drows = {}
                        for hh in (1.e-6, 5.e-7):
                            xm, xp_t = (torch.tensor([xp-v], dtype=torch.float64) for v in (hh, -hh))
                            zm = _binary_modes(xm / msec, m1, m2, s1, s2, fref, 0.)[(ell, m)][0]
                            zp = _binary_modes(xp_t / msec, m1, m2, s1, s2, fref, 0.)[(ell, m)][0]
                            assert torch.isfinite(zm.real) and torch.isfinite(zm.imag) and torch.abs(zm) > 0
                            assert torch.isfinite(zp.real) and torch.isfinite(zp.imag) and torch.abs(zp) > 0
                            pk = -mapped_phase_c1(torch.cat((xm, xp_t)), kmap_f0, kfrlm, kfr22,
                                                  m, positive_lm_kerr, ell=ell)
                            pd = float((pk[1] - pk[0]) / (2 * hh))
                            bd = float(torch.angle(zp * torch.conj(zm)) / (2 * hh))
                            drows[str(hh)] = {"binary_derivative": bd,
                                               "kerr_predictor_derivative": pd,
                                               "derivative_residual": bd - pd}
                        branch = "low" if xp < kmap_f0 else ("intermediate" if xp <= kfrlm else "high")
                        kerr_probe_rows.append({"x": xp, "branch": branch,
                            "phase_residual_rad": float(_wrapped(torch.angle(zbk) - predk)),
                            "derivatives": drows})
                else:
                    kerr_probe_rows = None
            else:
                k_low = k_mid = k_high = None
                kerr_probe_rows = None
            london_control = None
            if binary is not None and (2, 2) in london_qnm_modes and (ell, m) in london_qnm_modes:
                lfr22, lfd22 = london_qnm_modes[(2, 2)]
                lfrlm, lfdlm = london_qnm_modes[(ell, m)]
                lmap_f0 = .018 * lfrlm / lfr22
                positive_lm_london = lambda y: carrier0(y) + phenomd_eq14_phase_delta_c1(
                    y, eta, chi_pn, lfr22, lfrlm, lfd22, lfdlm, a4, a5,
                    qnm_ratio=lfr22 / lfrlm)
                london_pred = -mapped_phase_c1(
                    x, lmap_f0, lfrlm, lfr22, m, positive_lm_london, ell=ell) + gauge0
                l_low = valid & (x < lmap_f0)
                l_mid = valid & (x >= lmap_f0) & (x <= lfrlm)
                l_high = valid & (x > lfrlm)
                london_control = {
                    "map_f_rd_22": lfr22, "map_f_rd_lm": lfrlm,
                    "map_f0": lmap_f0,
                    "phase_map_transition_formula": ".018*f_rd_lm(London)/f_rd_22(London)",
                    "pn_amplitude_map_transition": {
                        "formula": ".014*f_rd_lm(London)/f_rd_22(London)",
                        "f0": .014*lfrlm/lfr22,
                        "frequency_functions": "London printed QNM fits for both 22 and lm"},
                    "eq14_f_damp_22": lfd22,
                    "eq14_f_damp_lm": lfdlm,
                    "binary_comparison": {
                        "fullband": raw_metrics(bmode, london_pred, valid),
                        "inspiral": raw_metrics(bmode, london_pred, l_low),
                        "intermediate": raw_metrics(bmode, london_pred, l_mid),
                        "merger_ringdown": raw_metrics(bmode, london_pred, l_high)},
                    "effect_vs_independent_kerr": {
                        "phase_rms_rad": float(torch.sqrt(torch.mean(_wrapped(london_pred-corrected018_kerr)[valid]**2))) if kerr_available else None,
                        "phase_max_rad": float(torch.max(torch.abs(_wrapped(london_pred-corrected018_kerr)[valid]))) if kerr_available else None}}
            mapped_carrier_isolation = None
            if kerr_available and (ell, m) in ((2, 1), (3, 3)):
                mapped_carrier_isolation = []
                mid_slope_k = (kfr22 - 2*kmap_f0/m) / (kfrlm-kmap_f0)
                for xp in (.02, .04, .06, .08, .12):
                    if xp <= kmap_f0:
                        g = 2*xp/m; kap = m/2.; branch = "low"
                    elif xp <= kfrlm:
                        g = mid_slope_k*(xp-kmap_f0) + 2*kmap_f0/m; kap = 1/mid_slope_k; branch = "intermediate"
                    else:
                        g = kfr22/kfrlm*xp; kap = kfrlm/kfr22; branch = "high"
                    terms = {}
                    for hh in (1.e-5, 5.e-7):
                        gm, gp = (torch.tensor([g-v], dtype=torch.float64) for v in (hh, -hh))
                        dm = d(gm/msec, m1,m2,s1,s2,phic=0.,tc=0.,f_ref=fref,return_phase=True)[2][0]
                        dp = d(gp/msec, m1,m2,s1,s2,phic=0.,tc=0.,f_ref=fref,return_phase=True)[2][0]
                        zm = _binary_modes(gm/msec,m1,m2,s1,s2,fref,0.)[(2,2)][0]
                        zp = _binary_modes(gp/msec,m1,m2,s1,s2,fref,0.)[(2,2)][0]
                        lm_m = _binary_modes(torch.tensor([(xp-hh)/msec],dtype=torch.float64),m1,m2,s1,s2,fref,0.)[(ell,m)][0]
                        lm_p = _binary_modes(torch.tensor([(xp+hh)/msec],dtype=torch.float64),m1,m2,s1,s2,fref,0.)[(ell,m)][0]
                        d22 = float(torch.angle(zp*torch.conj(zm))/(2*hh))
                        dD = float(-(dp-dm)/(2*hh))
                        raw_dd = float(phenomd_eq14_phase_delta_derivative(
                            torch.tensor([g],dtype=torch.float64), eta, chi_pn, fr22, kfrlm,
                            fd22, kfdlm, a4, a5, qnm_ratio=kfr22/kfrlm)[0])
                        raw_dj = float(phenomd_eq14_phase_delta_derivative(
                            torch.tensor([.5*fr22],dtype=torch.float64), eta, chi_pn, fr22, kfrlm,
                            fd22, kfdlm, a4, a5, qnm_ratio=kfr22/kfrlm)[0])
                        dc1 = 0. if g <= .5*fr22 else raw_dd - raw_dj
                        # kappa*g'=1 for every map branch. dD is already the
                        # physical (-p0) derivative; Eq14 is positive-p.
                        pred_native = dD - dc1
                        pred_oracle = d22 - dc1
                        qx = torch.tensor([xp-hh, xp+hh], dtype=torch.float64)
                        fullp = -mapped_phase_c1(qx, kmap_f0, kfrlm, kfr22, m,
                                                 positive_lm_kerr, ell=ell) + gauge0
                        full_cd = float((fullp[1]-fullp[0])/(2*hh))
                        actual = float(torch.angle(lm_p*torch.conj(lm_m))/(2*hh))
                        terms[str(hh)] = {"D_physical_carrier_derivative_at_g": dD,
                            "HM22_phasor_derivative_at_g": d22,
                            "Eq14_C1_delta_derivative_at_g": dc1,
                            "predicted_native_HMlm_derivative": pred_native,
                            "predicted_oracle_HMlm_derivative": pred_oracle,
                            "full_corrected018_kerr_central_derivative": full_cd,
                            "actual_HMlm_derivative": actual,
                            "carrier_error": d22-dD,
                            "native_derivative_residual": actual-pred_native,
                            "oracle_derivative_residual": actual-pred_oracle,
                            "full_central_check_residual": full_cd-pred_native}
                    assert abs(terms["1e-05"]["full_corrected018_kerr_central_derivative"] -
                               terms["5e-07"]["full_corrected018_kerr_central_derivative"]) < 1.e-2
                    mapped_carrier_isolation.append({"x":xp,"g":g,"branch":branch,"kappa":kap,"terms":terms})
            effective_map_inversion = None
            if binary is not None and kerr_available and (ell, m) in ((3, 3), (4, 4)):
                # In the intermediate branch Eq14 C1 is constant, and
                # kappa*g'=1, so d phi_lm/dx must equal the physical 22
                # derivative at the mapped g.  Numerically invert that
                # black-box derivative locally only as a diagnostic.
                def mode_derivative(mode, xx, hh=5.e-7):
                    zm = _binary_modes(torch.tensor([(xx-hh)/msec], dtype=torch.float64),
                                       m1,m2,s1,s2,fref,0.)[mode][0]
                    zp = _binary_modes(torch.tensor([(xx+hh)/msec], dtype=torch.float64),
                                       m1,m2,s1,s2,fref,0.)[mode][0]
                    assert torch.isfinite(zm.real) and torch.isfinite(zm.imag) and torch.abs(zm)>0
                    assert torch.isfinite(zp.real) and torch.isfinite(zp.imag) and torch.abs(zp)>0
                    return float(torch.angle(zp*torch.conj(zm))/(2*hh))
                inv_f0 = kmap_f0
                mid_slope = (kfr22 - 2*inv_f0/m) / (kfrlm-inv_f0)
                turn = inv_f0 + (.5*kfr22 - 2*inv_f0/m) / mid_slope
                points = [inv_f0 + q*(turn-inv_f0) for q in (.25, .5, .75)]
                inversion_stability = []
                for inv_h in (1.e-6, 5.e-7, 2.5e-7, 1.25e-7):
                    inv_rows = []
                    for xp in points:
                        pred_g = mid_slope*(xp-inv_f0) + 2*inv_f0/m
                        target = mode_derivative((ell,m), xp, inv_h)
                        span = max(.002, .25*pred_g)
                        lo, hi = max(1.e-5, pred_g-span), pred_g+span
                        flo = mode_derivative((2,2), lo, inv_h)-target
                        fhi = mode_derivative((2,2), hi, inv_h)-target
                        bracketed = flo*fhi <= 0.
                        if bracketed:
                            for _ in range(45):
                                midg = .5*(lo+hi)
                                fm = mode_derivative((2,2), midg, inv_h)-target
                                if flo*fm <= 0.: hi, fhi = midg, fm
                                else: lo, flo = midg, fm
                            eff_g = .5*(lo+hi)
                        else:
                            eff_g = None
                        inv_rows.append({"x":xp, "predicted_g":pred_g,
                            "target_lm_derivative":target, "bracket": [lo,hi],
                            "bracketed":bracketed, "effective_g":eff_g,
                            "effective_g_error":None if eff_g is None else eff_g-pred_g})
                    good = [(r["x"], r["effective_g"]) for r in inv_rows if r["effective_g"] is not None]
                    if len(good) >= 2:
                        sx = sum(q[0] for q in good); sy = sum(q[1] for q in good)
                        sxx = sum(q[0]*q[0] for q in good); sxy = sum(q[0]*q[1] for q in good)
                        slope = (len(good)*sxy-sx*sy)/(len(good)*sxx-sx*sx)
                        intercept = (sy-slope*sx)/len(good)
                        affine_res = max(abs(g-(intercept+slope*xx)) for xx,g in good)
                        implied_f0 = intercept/(2./m-slope)
                        implied_fr_lm = (kfr22-intercept)/slope
                        implied_onset = (.5*kfr22-intercept)/slope
                    else:
                        slope = intercept = affine_res = implied_f0 = implied_fr_lm = implied_onset = None
                    inversion_stability.append({"derivative_h": inv_h, "points": inv_rows,
                        "g_intercept": intercept, "g_slope": slope,
                        "max_abs_fit_residual": affine_res, "implied_f0": implied_f0,
                        "implied_f_rd_lm": implied_fr_lm, "implied_mr_onset_x": implied_onset})
                    if intercept is not None and slope is not None and abs(slope) > 0:
                        # Joint endpoint inference from the locally affine
                        # effective map.  This is a diagnostic calibration
                        # control only; these values never enter a predictor.
                        joint_f0 = intercept / (2. / m - slope)
                        joint_ratio = joint_f0 / .018
                        joint_fr22 = intercept / (1. - slope * joint_ratio)
                        joint_frlm = joint_ratio * joint_fr22
                    else:
                        joint_f0 = joint_fr22 = joint_frlm = None
                    inversion_stability[-1]["joint_inferred_endpoints"] = {
                        "f0": joint_f0, "f_rd_22": joint_fr22,
                        "f_rd_lm": joint_frlm,
                        "formula": "f0=b/(2/m-s), R=f0/.018, f22=b/(1-sR), flm=R*f22"}
                effective_map_inversion = {"map": "independent_kerr_018",
                    "derivative_stability": inversion_stability,
                    "literal_f0": inv_f0, "literal_frlm": kfrlm,
                    "literal_endpoint_g": .5*kfr22,
                    "literal_endpoint_x": turn,
                    "qnm_frequency_controls": {
                        "inferred_f_rd_lm_by_h": {str(r["derivative_h"]): r["implied_f_rd_lm"]
                                                   for r in inversion_stability},
                        "independent_kerr_f_rd_lm": kfrlm,
                        "public_hm_callable_f_rd_lm": frlm,
                        "public_lal_eob_f_rd_lm": {
                            name: (None if values.get((ell,m)) is None else values[(ell,m)][0])
                            for name, values in component_qnm.items()}}}
            component_controls = {}
            for cname, cq in component_qnm.items():
                if cq.get((2, 2)) is None or cq.get((ell, m)) is None:
                    component_controls[cname] = {"status": "unsupported_mode"}
                    continue
                cfr22, cfd22 = cq[(2, 2)]
                cfrlm, cfdlm = cq[(ell, m)]
                cpositive = lambda y: carrier0(y) + phenomd_eq14_phase_delta_c1(
                    y, eta, chi_pn, fr22, cfrlm, fd22, cfdlm, a4, a5,
                    qnm_ratio=cfr22 / cfrlm)
                cpred = -mapped_phase_c1(x, .018*cfrlm/cfr22, cfrlm, cfr22,
                                         m, cpositive, ell=ell) + gauge0
                component_controls[cname] = {"f_rd_22": cfr22, "f_damp_22": cfd22,
                    "f_rd_lm": cfrlm, "f_damp_lm": cfdlm,
                    "vs_independent_kerr": None if not kerr_available else {
                        "22_f_rd_difference": cfr22-kfr22, "lm_f_rd_difference": cfrlm-kfrlm,
                        "22_f_damp_difference": cfd22-kfd22, "lm_f_damp_difference": cfdlm-kfdlm},
                    "binary_comparison": None if binary is None else {
                        "fullband": raw_metrics(bmode, cpred, valid),
                        "lowband": raw_metrics(bmode, cpred, valid & (x < .018*cfrlm/cfr22)),
                        "intermediateband": raw_metrics(bmode, cpred, valid & (x >= .018*cfrlm/cfr22) & (x <= cfrlm)),
                        "highband": raw_metrics(bmode, cpred, valid & (x > cfrlm))}}
            # Compact onset probes in the native Mf coordinate.
            probe_x = sorted(set(float(v) for v in
                (.5*f0, .9*f0, f0, 1.01*f0, 1.1*f0, .25*frlm,
                 .5*frlm, frlm, 1.2*frlm)))
            probe_rows = []
            for xp in probe_x:
                fp = torch.tensor([xp / msec], dtype=torch.float64)
                zb = _binary_modes(fp, m1, m2, s1, s2, fref, 0.)[(ell,m)][0]
                xp_t = torch.tensor([xp], dtype=torch.float64)
                bp = -mapped_phase_c1(xp_t, f0, frlm, fr22, m, carrier0, ell=ell)[0] + gauge0
                ep = -mapped_phase_c1(xp_t, f0, frlm, fr22, m, positive_lm, ell=ell)[0] + gauge0
                deriv = {}
                for h in (1.e-6, 5.e-7):
                    vals = []
                    for carrier in (carrier0, positive_lm, positive_lm_hm_ratio):
                        q0 = torch.tensor([xp-h, xp+h], dtype=torch.float64)
                        yy = -mapped_phase_c1(q0, f0, frlm, fr22, m, carrier, ell=ell) + gauge0
                        zm = _binary_modes(torch.tensor([(xp-h)/msec], dtype=torch.float64), m1, m2, s1, s2, fref, 0.)[(ell,m)]
                        zp = _binary_modes(torch.tensor([(xp+h)/msec], dtype=torch.float64), m1, m2, s1, s2, fref, 0.)[(ell,m)]
                        assert zm.numel() == 1 and zp.numel() == 1
                        zz = torch.stack((zm[0], zp[0]))
                        vals.append((float((yy[1]-yy[0])/(2*h)), float(torch.angle(zz[1]*torch.conj(zz[0]))/(2*h))))
                    deriv[str(h)] = {"baseline_phase_derivative": vals[0][0], "eq14_phase_derivative": vals[1][0],
                                     "hm_ratio_eq14_phase_derivative": vals[2][0],
                                     "binary_phase_derivative": vals[0][1],
                                     "baseline_derivative_residual": vals[0][1]-vals[0][0],
                                     "eq14_derivative_residual": vals[0][1]-vals[1][0],
                                     "hm_ratio_eq14_derivative_residual": vals[0][1]-vals[2][0]}
                if (ell, m) == (2, 2) and abs(xp - .5*f0) < 1.e-14:
                    hh = 1.e-6
                    pp = d(torch.tensor([(xp+hh)/msec], dtype=torch.float64), m1, m2, s1, s2,
                            phic=0., tc=0., f_ref=fref, return_phase=True)[2][0]
                    pm = d(torch.tensor([(xp-hh)/msec], dtype=torch.float64), m1, m2, s1, s2,
                            phic=0., tc=0., f_ref=fref, return_phase=True)[2][0]
                    dcheck = float(-(pp-pm)/(2*hh))
                    assert abs((dcheck - deriv["1e-06"]["baseline_phase_derivative"]) / dcheck) < 1e-3
                    assert abs((deriv["1e-06"]["binary_phase_derivative"] - dcheck) / dcheck) < 1e-3
                    checks.append("binary 22 finite-difference derivative agrees with D")
                probe_rows.append({"x": xp, "frequency_hz": xp/msec,
                    "baseline_phase_residual_rad": float(_wrapped(torch.angle(zb)-bp)),
                    "eq14_phase_residual_rad": float(_wrapped(torch.angle(zb)-ep)),
                    "derivatives": deriv})
            residual = _wrapped(torch.angle(z) - eq14)
            design = torch.stack((torch.ones_like(f_hz), f_hz), dim=1)
            fit = torch.linalg.lstsq(design[valid], residual[valid]).solution
            low_controls = []
            for mult in (1.01, 1.1, 1.2, 1.28, 1.3, 1.4):
                xp = mult * f0; h = 1.e-6
                zl = _binary_modes(torch.tensor([xp/msec], dtype=torch.float64), m1,m2,s1,s2,fref,0.)[(ell,m)][0]
                assert torch.isfinite(zl.real) and torch.isfinite(zl.imag) and torch.abs(zl) > 0
                lowp = -(m/2.) * carrier0(torch.tensor([2*xp/m], dtype=torch.float64))[0] + gauge0
                lowp -= low_frequency_offset(ell, m)
                lowpm = -(m/2.) * carrier0(torch.tensor([2*(xp-h)/m], dtype=torch.float64))[0] + gauge0
                lowpp = -(m/2.) * carrier0(torch.tensor([2*(xp+h)/m], dtype=torch.float64))[0] + gauge0
                zm = _binary_modes(torch.tensor([(xp-h)/msec], dtype=torch.float64), m1,m2,s1,s2,fref,0.)[(ell,m)][0]
                zp = _binary_modes(torch.tensor([(xp+h)/msec], dtype=torch.float64), m1,m2,s1,s2,fref,0.)[(ell,m)][0]
                assert torch.isfinite(zm.real) and torch.isfinite(zm.imag) and torch.abs(zm) > 0
                assert torch.isfinite(zp.real) and torch.isfinite(zp.imag) and torch.abs(zp) > 0
                low_controls.append({"x_over_f0": mult, "phase_residual_rad": float(_wrapped(torch.angle(zl)-lowp)),
                    "binary_derivative": float(torch.angle(zp*torch.conj(zm))/(2*h)),
                    "low_branch_derivative": float((lowpp-lowpm)/(2*h))})
            binary_metrics = None if binary is None else {
                "raw_vs_native": compare(binary[(ell, m)]),
                "parity_adjusted_vs_native": compare(((-1) ** ell) * binary[(ell, m)]),
                "native22_l2": binary22_l2}
            mode_rows[f"{ell}{m}"] = {"qnm": {"f_rd": frlm, "f_damp": fdlm},
                "fullband": {"baseline": clean_metrics(base), "eq14": clean_metrics(eq14)},
                "corrected_physical_predictor": clean_metrics(corrected),
                "corrected_018_transition_control": clean_metrics(corrected018),
                "qnm_denominator_controls": {
                    "map_frequency_functions": {
                        "carrier": "D remnant_properties f_rd22/f_damp22",
                        "lm": "HM.mode_qnm_frequencies(l,m) except 22 uses HM 22",
                        "denominator": "HM.mode_qnm_frequencies(2,2)"
                    },
                    "corrected_014_hm22_binary_comparison": None if binary is None else raw_metrics(bmode, corrected014_hm22, valid),
                    "corrected_018_hm22_binary_comparison": None if binary is None else raw_metrics(bmode, corrected018_hm22, valid),
                    "corrected_018_hm22_ratio_binary_comparison": None if binary is None else {
                        "fullband": raw_metrics(bmode, corrected018_hm22_ratio, valid),
                        "lowband": raw_metrics(bmode, corrected018_hm22_ratio, low),
                        "intermediateband": raw_metrics(bmode, corrected018_hm22_ratio, mid),
                        "highband": raw_metrics(bmode, corrected018_hm22_ratio, high)},
                    "eq14_qnm_ratio": hm_ratio,
                    "independent_kerr_018": None if not kerr_available else {
                        "map_f_rd_22": kfr22, "map_f_rd_lm": kfrlm,
                        "map_f0": kmap_f0, "eq14_ratio": kfr22 / kfrlm,
                        "binary_comparison": None if binary is None else {
                            "fullband": raw_metrics(bmode, corrected018_kerr, valid),
                            "lowband": raw_metrics(bmode, corrected018_kerr, k_low),
                            "intermediateband": raw_metrics(bmode, corrected018_kerr, k_mid),
                            "highband": raw_metrics(bmode, corrected018_kerr, k_high)},
                        "compact_probes": kerr_probe_rows},
                    "london_published_018": london_control,
                    "public_eob_qnm_component_controls": component_controls,
                    "corrected_014_hm22_clean": clean_metrics(corrected014_hm22),
                    "corrected_018_hm22_clean": clean_metrics(corrected018_hm22)},
                "corrected_binary_comparison": None if binary is None else {
                    "fullband": raw_metrics(bmode, corrected_with_gauge, valid),
                    "lowband": raw_metrics(bmode, corrected_with_gauge, low),
                    "intermediateband": raw_metrics(bmode, corrected_with_gauge, mid),
                    "highband": raw_metrics(bmode, corrected_with_gauge, high)},
                "corrected_018_binary_comparison": None if binary is None else raw_metrics(bmode, corrected018, valid),
                "low_branch_controls": low_controls,
                "join_probes": probe_rows,
                "mapped_carrier_isolation": mapped_carrier_isolation,
                "effective_map_inversion": effective_map_inversion,
                "lowband": {"baseline": clean_metrics(base, low), "eq14": clean_metrics(eq14, low)},
                "intermediateband": {"baseline": clean_metrics(base, mid), "eq14": clean_metrics(eq14, mid)},
                "highband": {"baseline": clean_metrics(base, high), "eq14": clean_metrics(eq14, high)},
                "amplitude_join_018_control": clean_metrics(amp_join_control),
                "residual_structure_diagnostic": {"wrapped_residual_mean_rad": float(torch.mean(residual[valid])),
                    "wrapped_residual_linear_slope_rad_per_hz": float(fit[1]),
                    "wrapped_residual_linear_intercept_rad": float(fit[0])},
                "binary_vs_native": binary_metrics,
                "binary_vs_clean": None if binary is None else {"baseline": binary_clean(base), "eq14": binary_clean(eq14)},
                "eq14_delta_rms_rad": float(torch.sqrt(torch.mean((eq14-base)**2))),
                "start22_convention": convention}
        # Reference experiment: at f=m*f_ref/2, the low map samples the
        # carrier at f_ref.  No phase/slope alignment is applied here.
        reference_experiment = []
        exact_corrected_residuals = []
        for rf in (20., 30., 40.):
            for ph in (0., .2, .7):
                ref_rows = {}
                for ell, m in MODES:
                    target = torch.tensor([m * rf / 2.], dtype=torch.float64)
                    zref = _binary_modes(target, m1, m2, s1, s2, rf, ph)[(ell, m)][0]
                    def cref(y):
                        return -d(y / msec, m1, m2, s1, s2, phic=0., tc=0., f_ref=rf,
                                  return_phase=True)[2]
                    xx = target * msec
                    qn = mode_rows[f"{ell}{m}"]["qnm"]
                    qfr, qfd = qn["f_rd"], qn["f_damp"]
                    pred = mapped_phase_c1(xx, .014 * qfr / fr22, qfr, fr22, m, cref, ell=ell)[0] + m * ph
                    p0ref = float(-cref(torch.tensor([rf * msec], dtype=torch.float64))[0])
                    assert abs(p0ref) < 1e-9
                    delta = low_frequency_offset(ell, m)
                    pmap = mapped_phase_c1(xx, .014 * qfr / fr22, qfr, fr22, m,
                                           lambda y: -cref(y), ell=ell)[0] - m * ph
                    gauge = (m / 2. - 1.) * slope_peak * msec * rf
                    corrected = _wrapped(torch.angle(zref) - (-pmap + gauge))
                    exact_corrected_residuals.append(float(abs(corrected)))
                    ref_rows[f"{ell}{m}"] = {"target_hz": float(target[0]),
                        "p0_at_fref": p0ref,
                        "observed_phase_rad": float(torch.angle(zref)),
                        "observed_parity_adjusted_phase_rad": float(torch.angle(((-1)**ell)*zref)),
                        "paper_delta_rad": float(delta),
                        "residual_to_delta_parity_adjusted_rad": float(_wrapped(torch.angle(((-1)**ell)*zref) - delta)),
                        "predicted_clean_phase_rad": float(pred),
                        "analytic_time_gauge_correction_rad": gauge,
                        "residual_after_analytic_time_gauge_rad": float(_wrapped(torch.angle(zref) - pred - gauge)),
                        "residual_corrected_physical_rad": float(corrected),
                        "positive_p_representation_phase_rad": float(pmap),
                        "positive_p_representation_phasor_phase_rad": float(torch.angle(torch.exp(-1j * pmap)))}
                reference_experiment.append({"f_ref_hz": rf, "phic": ph, "modes": ref_rows})
        fixed_probe = []
        for rf in (20., 30., 40.):
            ff = torch.tensor([40.], dtype=torch.float64)
            bz = _binary_modes(ff, m1, m2, s1, s2, rf, 0.)
            fixed_probe.append({"f_hz": 40., "f_ref_hz": rf,
                "modes_phase_rad": {f"{l}{m}": float(torch.angle(bz[(l,m)][0])) for l,m in MODES}})
        worst_ref = max((abs(v["residual_corrected_physical_rad"]), rf, ph, k, v["target_hz"])
                        for ent in reference_experiment for rf, ph in [(ent["f_ref_hz"], ent["phic"])]
                        for k, v in ent["modes"].items())
        rows.append({"mass1": m1, "mass2": m2, "chi1": s1, "chi2": s2,
                     "f_ref_hz": fref, "Msec": msec, "eta": eta,
                     "chi_pn_unscaled": chi_pn, "slope_peak_rad_per_Mf": slope_peak, "remnant_Mf_Msun": mf,
                     "remnant_af": af, "qnm_22": {"f_rd": fr22, "f_damp": fd22},
                     "hm_qnm_22": {"f_rd": hm_fr22, "f_damp": hm_fd22},
                     "independent_kerr_qnm_22": {"f_rd": kerr_qnm[(2, 2)][0], "f_damp": kerr_qnm[(2, 2)][1],
                                                   "conversion": "Re/Im(Mf omega)/(2pi)*(M/Mf)"},
                     "london_published_qnm": {f"{l}{mm}": {"f_rd": v[0], "f_damp": v[1]}
                                               for (l, mm), v in london_qnm_modes.items()},
                     "modes": mode_rows, "binary_error": binary_error,
                     "reference_experiment": reference_experiment,
                     "exact_corrected_max_abs_rad": max(exact_corrected_residuals),
                     "exact_corrected_assert_lt_1e-4": max(exact_corrected_residuals) < 1e-4,
                     "exact_corrected_worst": {"abs_rad": worst_ref[0], "f_ref_hz": worst_ref[1],
                        "phic": worst_ref[2], "mode": worst_ref[3], "target_hz": worst_ref[4]},
                     "fixed_physical_probe": fixed_probe,
                     "binary_modes": sorted([f"{l}{m}" for l, m in binary]) if binary else None})
    here = pathlib.Path(__file__)
    clean = pathlib.Path(inspect.getfile(mapped_phase_c1))
    out = {"provenance": {"papers": ["https://arxiv.org/html/1708.00404v4 Eq. 3,5,8,9,13,14", "https://arxiv.org/html/1508.07253 Sec. V", "https://arxiv.org/html/1810.03550 Appendix A Eq. 21-32"],
        "units": "x=Msec*f_Hz; Msec=(m1+m2)*4.925490947641267e-6; physical phi22=-D p0; Eq14 is inside carrier before mapping",
        "phase_convention": "London Eq2 h_lm=A exp(+i phi_lm), so clean phasor is exp(+i mapped phase); native storage parity is reported separately",
        "joins": {"phase": ".014*f_rd_lm/f_rd_22 (D carrier denominator)",
                  "amplitude_control": ".018*f_rd_lm/f_rd_22",
                  "qnm_denominator_controls": {
                      "014_hm22": ".014*f_rd_lm/HM.mode_qnm_frequencies(2,2)",
                      "018_hm22": ".018*f_rd_lm/HM.mode_qnm_frequencies(2,2)"}},
        "independent_kerr_control": "kerr_qnm_generated.dimensionless_qnm(l,m,0,af), converted with (M/Mf)/(2pi); D carrier center/damping retained",
        "london_published_control": "Fresh transcription of London & Fauchon-Jones Appendix A Eq.21-32; London QNM 22 and lm endpoints replace mapping and Eq14 controls only; D carrier remains D",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "lalsimulation": getattr(lalsimulation, "__version__", "unknown"),
                     "lal": getattr(lal, "__version__", "unknown")},
        "source_sha256": {"script": _hash(here), "clean_phase": _hash(clean),
                          "clean_hm_map": _hash(pathlib.Path(inspect.getfile(mapped_phase_c1))),
                          "binary_api_doc": hashlib.sha256((lalsimulation.SimIMRPhenomHMGethlmModes.__doc__ or '').encode()).hexdigest()},
        "assertions": checks, "native_api": str(inspect.signature(IMRPhenomHM.__call__))},
        "grid_hz": {"start": 20., "stop": 512., "count": len(f_hz)}, "cases": rows}
    pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(pathlib.Path(__file__).with_suffix('.json'))


if __name__ == '__main__':
    main()
