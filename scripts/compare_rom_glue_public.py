#!/usr/bin/env python3
"""Five full-grid ROM comparisons against the public SEOBNRv4_ROM API.

The independent QNM column uses only the clean published remnant fit and
independent Kerr 220 evaluator; it is a diagnostic control, never a runtime
oracle or fitted replacement.
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import lal
import lalsimulation as ls
import numpy as np
import torch

from torchwave.approximants.rom.bspline import load_official_rom
from torchwave.approximants.rom.glue import OfficialROMGlue
from torchwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from parity_harness import get_canonical_catalog_cases

ROM = Path('/private/tmp/torchwave-oracle-data/SEOBNRv4ROM_v3.0.hdf5')
MSUN_SEC = 4.925490947641267e-6
APP = ls.GetApproximantFromString('SEOBNRv4_ROM')

# Four predeclared controls exercise nonzero inclination and phic, with one
# fref=0 control (the wrapper's documented effective-lower sentinel).
INDEPENDENT = [
    dict(id='independent_a', mass1=50., mass2=25., chi1=.2, chi2=.1,
         distance=500., inclination=.73, phic=.17, tc=0., fref=20., df=.5, flow=20., fhigh=256.),
    dict(id='independent_b', mass1=35., mass2=35., chi1=-.4, chi2=.3,
         distance=500., inclination=.61, phic=.23, tc=0., fref=30., df=.5, flow=20., fhigh=256.),
    dict(id='independent_c', mass1=23., mass2=21., chi1=-.6, chi2=.2,
         distance=680., inclination=.9, phic=.71, tc=0., fref=35., df=.25, flow=20., fhigh=512.),
    dict(id='independent_d', mass1=80., mass2=20., chi1=.6, chi2=-.2,
         distance=400., inclination=.44, phic=.31, tc=0., fref=0., df=.5, flow=20., fhigh=256.),
]

def canonical_case():
    for c in get_canonical_catalog_cases():
        if c.get('torch_name') == 'SEOBNR_ROM':
            p, g = c['params'], c['grid']
            return dict(id=c['id'], mass1=p['mass1'], mass2=p['mass2'], chi1=p['chi1'], chi2=p['chi2'],
                        distance=p['distance'], inclination=p.get('inclination', 0.), phic=p.get('phic', 0.),
                        tc=p.get('tc', 0.), fref=20., df=g['df'], flow=g['flow'], fhigh=g['fhigh'])
    raise RuntimeError('SEOBNR_ROM canonical case missing')

def public_waveform(c):
    # fref=0 has the same public effective reference as flow for this API.
    fref = c['flow'] if c['fref'] == 0 else c['fref']
    hp, hc = ls.SimInspiralChooseFDWaveform(
        c['mass1']*lal.MSUN_SI, c['mass2']*lal.MSUN_SI,
        0., 0., c['chi1'], 0., 0., c['chi2'],
        c['distance']*1e6*lal.PC_SI, c['inclination'], c['phic'],
        0., 0., 0., c['df'], c['flow'], c['fhigh'], fref, None, APP)
    f = np.arange(hp.data.length, dtype=np.float64) * hp.deltaF
    return f, hp.data.data.copy(), hc.data.data.copy()

def public_rd(c):
    v = lal.CreateCOMPLEX16Vector(1)
    ls.SimIMREOBGenerateQNMFreqV2(v, c['mass1'], c['mass2'], [0.,0.,c['chi1']], [0.,0.,c['chi2']], 2, 2, 1, ls.GetApproximantFromString('SEOBNRv4'))
    return float(v.data[0].real * (c['mass1']+c['mass2']) * MSUN_SEC / (2*np.pi))

def independent_rd(c):
    t = lambda x: torch.tensor(float(x), dtype=torch.float64)
    mf = eob_final_mass_fraction(t(c['mass1']), t(c['mass2']), t(c['chi1']), t(c['chi2']))
    af = hofmann_final_spin(t(c['mass1']), t(c['mass2']), t(c['chi1']), t(c['chi2']), fit_order=(3,4))
    qreal = signed_qnm_220(af)[0]
    # qreal=M_final*omega; ROM tables require total-M*f_RD in cycles.
    rd = qreal / (2*torch.pi*mf)
    return float(rd), float(mf), float(af), float(qreal)

def piecewise_rd(c):
    t = lambda x: torch.tensor(float(x), dtype=torch.float64)
    mf = eob_final_mass_fraction(t(c['mass1']), t(c['mass2']), t(c['chi1']), t(c['chi2']))
    af = hofmann_final_spin(t(c['mass1']), t(c['mass2']), t(c['chi1']), t(c['chi2']), fit_order=(3,4))
    q = signed_qnm_220_table_convention(af)[0]
    return float(q / (2*torch.pi*mf)), {'spin': float(af), 'qreal_table_convention': float(q)}

def metric(a, b):
    a, b = np.asarray(a), np.asarray(b)
    finite = np.isfinite(a.real)&np.isfinite(a.imag)&np.isfinite(b.real)&np.isfinite(b.imag)
    if not finite.all(): raise RuntimeError('nonfinite waveform sample')
    nz_a, nz_b = np.abs(a)>0, np.abs(b)>0
    common = nz_a & nz_b
    den = np.linalg.norm(b)
    magden = np.linalg.norm(np.abs(b))
    phase_l2 = np.linalg.norm((a[common]/np.abs(a[common]))-(b[common]/np.abs(b[common]))) if np.any(common) else 0.
    return dict(complex_relative_l2=float(np.linalg.norm(a-b)/den) if den else 0.,
                magnitude_relative_l2=float(np.linalg.norm(np.abs(a)-np.abs(b))/magden) if magden else 0.,
                unit_phase_l2=float(phase_l2/np.sqrt(np.count_nonzero(common))) if np.any(common) else None,
                support_a=int(np.count_nonzero(nz_a)), support_b=int(np.count_nonzero(nz_b)),
                support_mismatch=int(np.count_nonzero(nz_a != nz_b)), common_support=int(np.count_nonzero(common)))

def run_variant(lo, hi, c, f, rd):
    g = OfficialROMGlue(lo, hi, ringdown_mf=rd, f_lower_hz=c['flow'], f_final_hz=c['fhigh'])
    ft = torch.tensor(f, dtype=torch.float64)
    hp, hc = g(ft, c['mass1'], c['mass2'], c['chi1'], c['chi2'], distance_mpc=c['distance'],
               inclination=c['inclination'], phic=c['phic'], fref_hz=c['fref'], tc=c['tc'])
    return hp.detach().cpu().numpy(), hc.detach().cpu().numpy()

def main():
    os.environ['LAL_DATA_PATH'] = str(ROM.parent)
    lo, hi = load_official_rom(ROM, 'sub1'), load_official_rom(ROM, 'sub2')
    cases = [canonical_case(), *INDEPENDENT]
    rows=[]
    for c in cases:
        f, hp_ref, hc_ref = public_waveform(c)
        rd_pub = public_rd(c)
        rd_ind, mf, af, qreal = independent_rd(c)
        rd_piece, piece_info = piecewise_rd(c)
        hp_p, hc_p = run_variant(lo, hi, c, f, rd_pub)
        hp_i, hc_i = run_variant(lo, hi, c, f, rd_ind)
        hp_g, hc_g = run_variant(lo, hi, c, f, rd_piece)
        rows.append(dict(case=c, count=len(f), df_hz=float(f[1]-f[0]),
                          public_qnm_mf=float(rd_pub), independent_qnm_mf=float(rd_ind),
                          independent_final_mass_fraction=mf, independent_final_spin=af,
                          independent_mfinal_omega=qreal,
                          piecewise_qnm_mf=rd_piece, piecewise_info=piece_info,
                          public=dict(hp=metric(hp_p,hp_ref), hc=metric(hc_p,hc_ref)),
                          independent=dict(hp=metric(hp_i,hp_ref), hc=metric(hc_i,hc_ref)),
                          piecewise=dict(hp=metric(hp_g,hp_ref), hc=metric(hc_g,hc_ref))))
    out={'source':{'rom':str(ROM), 'rom_sha256':hashlib.sha256(ROM.read_bytes()).hexdigest(),
                   'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'python':sys.version, 'torch':torch.__version__,
                   'lal':getattr(lal,'__version__','unknown'), 'lalsimulation':getattr(ls,'__version__','unknown'),
                   'independent_qnm_formula':'M_final*omega / (2*pi*final_mass_fraction) = total-M*f_RD'}, 'cases':rows}
    Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(out, indent=2))
if __name__=='__main__': main()
