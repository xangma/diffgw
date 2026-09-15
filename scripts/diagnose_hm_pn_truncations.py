"""Clean-room, one-mode PN truncation substitutions for IMRPhenomHM.

Only the in-memory h_hat_multipole dispatch is replaced.  The native callable
is restored in finally; this script never edits production source.
"""
import hashlib, inspect, json, pathlib
import numpy as np
import torch
import lal, lalsimulation as lalsim
from torchwave.approximants import imrphenom_hm as native
from torchwave.waveforms import get_fd_waveform

MODES=((2,2),(2,1),(3,3),(3,2),(4,4),(4,3))
CASES=(
 {'id':'catalog','mass1':40.,'mass2':15.,'chi1':.3,'chi2':-.2,'distance':500.,'inclination':.8,'phic':.2,'f_ref':20.,'flow':20.,'fhigh':512.,'df':.25},
 {'id':'holdout','mass1':55.,'mass2':11.,'chi1':-.4,'chi2':.2,'distance':680.,'inclination':1.1,'phic':.71,'f_ref':27.,'flow':20.,'fhigh':512.,'df':.25})

def binary(c, active):
    ma=lalsim.SimInspiralCreateModeArray()
    for l,m in active: lalsim.SimInspiralModeArrayActivateMode(ma,l,m)
    d=lal.CreateDict(); lalsim.SimInspiralWaveformParamsInsertModeArray(d,ma)
    return lalsim.SimInspiralChooseFDWaveform(c['mass1']*lal.MSUN_SI,c['mass2']*lal.MSUN_SI,0,0,c['chi1'],0,0,c['chi2'],c['distance']*1e6*lal.PC_SI,c['inclination'],c['phic'],0,0,0,c['df'],c['flow'],c['fhigh'],c['f_ref'],d,lalsim.GetApproximantFromString('IMRPhenomHM'))

def metric(x,y):
    return {'raw_l2':float(np.linalg.norm(x-y)/np.linalg.norm(y)), 'magnitude_l2':float(np.linalg.norm(np.abs(x)-np.abs(y))/np.linalg.norm(np.abs(y)))}

def truncation(name, v, eta, delta, cs, ca):
    """Mishra 1601.05588 Eq. (12b,d), with prescribed truncations."""
    if name=='21_leading': return -(2**.5/3)*delta*v
    if name=='21_v2spin':
        c=-(2**.5/3); return c*(delta*v-1.5*(ca+delta*cs)*v**2)
    if name=='21_v3':
        c=-(2**.5/3); return c*(delta*v-1.5*(ca+delta*cs)*v**2+delta*(335/672+117*eta/56)*v**3)
    if name=='32_leading': return -(1/3)*(5/7)**.5*(1-3*eta)*v**2
    if name=='43_leading': return -(3/4)*(3/35)**.5*delta*(1-2*eta)*v**3
    raise ValueError(name)

def run_case(c, variant):
    old=native.h_hat_multipole; calls={'count':0}
    target,choice=variant
    eta=(c['mass1']*c['mass2'])/(c['mass1']+c['mass2'])**2; delta=(c['mass1']-c['mass2'])/(c['mass1']+c['mass2']); cs=(c['chi1']+c['chi2'])/2; ca=(c['chi1']-c['chi2'])/2
    def replacement(l,m,v_m,delta0,eta0,cs0,ca0):
        calls['count']+=1
        if (l,m)!=target: return old(l,m,v_m,delta0,eta0,cs0,ca0)
        return truncation(choice,v_m,eta0,delta0,cs0,ca0)
    native.h_hat_multipole=replacement
    try:
        f=torch.arange(int(round(c['fhigh']/c['df']))+1,dtype=torch.float64)*c['df']; k=int(round(c['flow']/c['df']))
        out={}
        for label,active in [('mode', (target,)),('all',MODES)]:
            h,hc=get_fd_waveform(approximant='IMRPhenomHM',mass1=c['mass1'],mass2=c['mass2'],chi1=c['chi1'],chi2=c['chi2'],distance=c['distance'],inclination=c['inclination'],phic=c['phic'],tc=0.,f_ref=c['f_ref'],f_lower=c['flow'],f_final=c['fhigh'],sample_frequencies=f,modes=active)
            bh,bc=binary(c,active); hp=h.detach().numpy()[k:]; cp=hc.detach().numpy()[k:]; bp=bh.data.data[k:]; bc0=bc.data.data[k:]
            low=(f.numpy()[k:]>=20)&(f.numpy()[k:]<40)
            out[label]={'full':{'hp':metric(hp,bp),'hc':metric(cp,bc0)},'20_40Hz':{'hp':metric(hp[low],bp[low]),'hc':metric(cp[low],bc0[low])}}
        out['inputs_float64']={'eta':eta,'delta':delta,'chi_s':cs,'chi_a':ca}
        out['calls']=calls['count']; return out
    finally: native.h_hat_multipole=old

def run():
    path=pathlib.Path(native.__file__); report={'signature':str(inspect.signature(native.h_hat_multipole)),'equations':'Mishra arXiv:1601.05588 Eq. (12b) H21 and Eq. (12d) H32; Eq. (12g) H43 leading V^3. These are documented truncation experiments, not inferred compatibility choices.','native_source_sha256_before':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':[],'versions':{'lal':lal.__version__,'lalsimulation':lalsim.__version__}}
    variants=(('21_leading',(2,1),'21_leading'),('21_v2spin',(2,1),'21_v2spin'),('21_v3',(2,1),'21_v3'),('32_leading',(3,2),'32_leading'),('43_leading',(4,3),'43_leading'))
    for c in CASES:
        report['cases'].append({'id':c['id'],'inputs':c,'variants':{name:run_case(c,(target,choice)) for name,target,choice in variants}})
    report['native_source_sha256_after']=hashlib.sha256(path.read_bytes()).hexdigest(); p=pathlib.Path(__file__); report['script_sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': run()
