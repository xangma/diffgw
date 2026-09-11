"""Public waveform-output diagnostics only; no phase/time fitting or runtime calibration."""
import argparse, importlib.util, json
from pathlib import Path
import lal, lalsimulation as ls, numpy as np, torch
from torchwave.approximants.imrphenomd import IMRPhenomD

p=argparse.ArgumentParser();p.add_argument('--baseline-file',type=Path);p.add_argument('--json-out',type=Path,required=True);args=p.parse_args()
if args.baseline_file:
    spec=importlib.util.spec_from_file_location('baseline',args.baseline_file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
else:
    from torchwave.approximants import imrphenomd_nrtidal as mod
f=np.arange(30.,1800.25,.25);seq=lal.CreateREAL8Sequence(len(f));seq.data[:]=f
rows=[]
for case in [(1.4,1.3,0.,0.,350.,420.),(1.4,1.3,.05,-.02,350.,420.),(1.2,1.6,.02,-.03,800.,200.),(1.4,1.4,0.,0.,1000.,1000.)]:
    m1,m2,c1,c2,l1,l2=case
    def oracle(name):
        d=lal.CreateDict()
        if name!='IMRPhenomD':
            ls.SimInspiralWaveformParamsInsertTidalLambda1(d,l1)
            ls.SimInspiralWaveformParamsInsertTidalLambda2(d,l2)
        hp,_=ls.SimInspiralChooseFDWaveformSequence(.2,m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,c1,0.,0.,c2,30.,100e6*lal.PC_SI,.4,d,ls.GetApproximantFromString(name),seq)
        return hp.data.data.copy()
    kw=dict(f=torch.tensor(f),mass1=m1,mass2=m2,chi1=c1,chi2=c2,distance=100.,inclination=.4,phic=.2,f_ref=30.)
    base=IMRPhenomD()(**kw)[0].detach().numpy(); obase=oracle('IMRPhenomD')
    for name in ('IMRPhenomD_NRTidal','IMRPhenomD_NRTidalv2'):
        actual=getattr(mod,name)()(**kw,lambda1=l1,lambda2=l2)[0].detach().numpy();expected=oracle(name)
        tr=actual/base;lr=expected/obase
        def relative_norm(x,y):return float(np.linalg.norm(x-y)/np.linalg.norm(y))
        mask=(f<=1000)&(abs(lr)>1e-10)
        row=dict(model=name,params=case,tidal_factor_l2=relative_norm(tr,lr),tidal_factor_amplitude_l2=relative_norm(abs(tr),abs(lr)),sub1000_factor_l2=relative_norm(tr[mask],lr[mask]),raw_complex_l2=relative_norm(actual,expected),phase_error_at_30=float(np.angle(tr[0]/lr[0])),phase_error_at_1000=float(np.angle(tr[np.where(f==1000)[0][0]]/lr[np.where(f==1000)[0][0]])))
        rows.append(row)
args.json_out.write_text(json.dumps({'note':'External public-output diagnostics. Ratios isolate tidal modifications but are not full waveform acceptance. No fitted alignment.','baseline':bool(args.baseline_file),'rows':rows},indent=2)+'\n')
for r in rows:print(json.dumps(r))
