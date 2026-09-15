"""Record the public LAL SEOBNRv4 TD phiRef/f_ref phase convention."""
import argparse, hashlib, json
from pathlib import Path
import lal, lalsimulation as lalsim
import numpy as np
def _invert_22(plus, cross, inclination):
    c=np.cos(inclination)
    return np.sqrt(16*np.pi/5)/(1+c*c)*plus-1j*np.sqrt(4*np.pi/5)/c*cross

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    m1,m2,inc,distance,dt,flow=30.,20.,.8,100e6*lal.PC_SI,1/4096,20.
    app=lalsim.GetApproximantFromString('SEOBNRv4'); rows=[]
    for fref in (0.,30.):
      for phi in (0.,.37,.91):
        hp,hc=lalsim.SimInspiralChooseTDWaveform(m1*lal.MSUN_SI,m2*lal.MSUN_SI,0,0,0,0,0,0,distance,inc,phi,0,0,0,dt,flow,fref,lal.CreateDict(),app)
        t=float(hp.epoch)+np.arange(len(hp.data.data))*float(hp.deltaT); z=_invert_22(np.asarray(hp.data.data),np.asarray(hc.data.data),inc)
        active=np.abs(z)>.001*np.max(np.abs(z)); rows.append(dict(f_ref=fref,phiRef=phi,n=len(z),epoch=float(hp.epoch),end=float(t[-1]),peak_time=float(t[np.argmax(abs(z))]),first_nonzero_phase=float(np.angle(z[np.flatnonzero(active)[0]])),peak_phase=float(np.angle(z[np.argmax(abs(z))])),phase_evolution=[float(np.unwrap(np.angle(z))[i]) for i in (np.flatnonzero(active)[0],len(z)//2,np.argmax(abs(z)))],z=z,t=t))
    base={(r['f_ref'],r['phiRef']):r for r in rows}; out=dict(api='SimInspiralChooseTDWaveform',approximant='SEOBNRv4',parameters=dict(mass1=m1,mass2=m2,inclination=inc,distance_mpc=100,deltaT=dt,f_start=flow),rows=[])
    for r in rows:
      q={k:v for k,v in r.items() if k not in ('z','t')}; ref=base[(r['f_ref'],0.)]; active=np.abs(ref['z'])>.001*np.max(np.abs(ref['z'])); ratio=r['z'][active]/ref['z'][active]; expected=np.exp(-2j*r['phiRef']); q['rotation_max_abs_error']=float(np.max(np.abs(ratio/expected-1))); out['rows'].append(q)
    out['source_sha256']={str(x):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in ('scripts/verify_seobnrv4_td_phase_sweep.py','scripts/verify_eob_published_waveforms.py')}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
