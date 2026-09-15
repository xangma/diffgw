"""Black-box Pv2 parity report for the clean driver (requires LALSuite)."""
import json, pathlib, hashlib
import numpy as np
import torch
from torchwave.approximants.phenompv2_waveform_clean import waveform

CASES = [
    dict(label='frozen_catalog', mass1=35., mass2=25., spin1x=.2, spin1y=.1, spin1z=.2, spin2x=-.1, spin2y=.15, spin2z=-.1, inclination=.7, phic=.3),
    dict(label='frozen_holdout', mass1=42., mass2=17., spin1x=.3, spin1y=-.15, spin1z=-.3, spin2x=.12, spin2y=.08, spin2z=.2, inclination=1.1, phic=.71, f_ref=35.),
    dict(label='independent_low_mass', mass1=20., mass2=15., spin1x=.1, spin1y=.1, spin1z=.2, spin2x=0., spin2y=0., spin2z=0., inclination=.2, phic=0.),
    dict(label='independent_two_spin_a', mass1=20., mass2=10., spin1x=.35, spin1y=-.15, spin1z=.1, spin2x=.1, spin2y=.2, spin2z=-.25, inclination=1.1, phic=-.2),
    dict(label='independent_two_spin_b', mass1=12., mass2=11., spin1x=-.2, spin1y=.25, spin1z=-.1, spin2x=.3, spin2y=.05, spin2z=.2, inclination=.4, phic=.8),
]

def main():
    outdir=pathlib.Path('docs/reviews/2026-09-13-parity-review/pv2-clean'); outdir.mkdir(parents=True, exist_ok=True)
    f=torch.arange(0.,512.0001,.25,dtype=torch.float64); rows=[]
    try:
        import lalsimulation as ls
        import lal
    except Exception as e:
        report={'status':'unavailable','reason':str(e),'cases':[]}
    else:
        for case in CASES:
            p,c=waveform(f, distance=500., tc=0., f_ref=case.get('f_ref',20.), **{k:v for k,v in case.items() if k not in ('label','f_ref')})
            rf=case.get('f_ref',20.)
            hp,hc=ls.SimInspiralChooseFDWaveform(case['mass1']*lal.MSUN_SI,case['mass2']*lal.MSUN_SI,case['spin1x'],case['spin1y'],case['spin1z'],case['spin2x'],case['spin2y'],case['spin2z'],500.*1e6*lal.PC_SI,case['inclination'],case['phic'],0,0,0,.25,20.,512.,rf,lal.CreateDict(),ls.IMRPhenomPv2)
            hp_o=np.asarray(hp.data.data).copy(); hc_o=np.asarray(hc.data.data).copy()
            t=(p.detach().numpy()+1j*c.detach().numpy()); mask=np.arange(len(t))*.25>=20
            hp_t, hc_t = p.detach().numpy()[mask].copy(), c.detach().numpy()[mask].copy()
            hp_q, hc_q = hp_o[:len(mask)][mask].copy(), hc_o[:len(mask)][mask].copy()
            def metrics(a,b):
                scale=max(float(np.max(np.abs(b))),1e-300)
                return {'relative_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)),
                        'max_normalized_error':float(np.max(np.abs(a-b))/scale)}
            # A fixed detector projection makes polarization-basis errors
            # visible without combining hp and hc into a cancellation-prone norm.
            Fp,Fc=.7,-.2
            det_t=Fp*hp_t+Fc*hc_t; det_q=Fp*hp_q+Fc*hc_q
            rows.append({'label':case['label'],'parameters':{k:v for k,v in case.items() if k != 'label'},'hp':{'full':metrics(hp_t,hp_q),'interior':metrics(hp_t[:-1],hp_q[:-1]),'max_error_frequency_hz':float(20.+np.argmax(np.abs(hp_t-hp_q))*.25)},'hc':{'full':metrics(hc_t,hc_q),'interior':metrics(hc_t[:-1],hc_q[:-1]),'max_error_frequency_hz':float(20.+np.argmax(np.abs(hc_t-hc_q))*.25)},
                         'detector':{'full':metrics(det_t,det_q),'interior':metrics(det_t[:-1],det_q[:-1]),'max_error_frequency_hz':float(20.+np.argmax(np.abs(det_t-det_q))*.25)}})
        report={'status':'ok','cases':rows}
    report.update({'provenance':'clean-room public Pv2 geometry, 2PN L, NNLO angles; D carrier callable; no fitted residuals','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
    (outdir/'diagnostic.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
