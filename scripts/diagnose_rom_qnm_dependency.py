#!/usr/bin/env python3
"""Compare published clean remnant/Kerr-QNM inputs with public SEOBNRv4 QNM."""
from __future__ import annotations
import hashlib,json,sys,os
from pathlib import Path
import lal,lalsimulation as ls,numpy as np,torch
from torchwave.constants import MSUN_SEC
from torchwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from torchwave.approximants.kerr_qnm_london_published import dimensionless_qnm as london_qnm
from torchwave.approximants.rom.bspline import evaluate_stitched_frequency,load_official_rom

ROM=Path('/private/tmp/torchwave-oracle-data/SEOBNRv4ROM_v3.0.hdf5'); PI=np.pi
CASES=[(50.,25.,.2,.1),(35.,35.,-.4,.3),(80.,20.,.6,-.2),(42.,18.,-.3,.45)]
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def public(case):
 m1,m2,s1,s2=case; hp=ls.SimInspiralChooseFDWaveform(m1*lal.MSUN_SI,m2*lal.MSUN_SI,0,0,s1,0,0,s2,500*lal.PC_SI*1e6,0,0,0,0,0,1,20,512,0,None,ls.GetApproximantFromString('SEOBNRv4_ROM'))[0]; return np.arange(hp.data.length),hp.data.data.copy()
def public_qnm(c):
 q=lal.CreateCOMPLEX16Vector(1); ls.SimIMREOBGenerateQNMFreqV2(q,*c[:2],[0,0,c[2]],[0,0,c[3]],2,2,1,ls.GetApproximantFromString('SEOBNRv4')); return float(q.data[0].real*sum(c[:2])*MSUN_SEC/(2*PI))
def metric(z,r):
 v=np.isfinite(z)&np.isfinite(r)&(np.abs(z)>0)&(np.abs(r)>0); return {'support':int(v.sum()),'support_mismatch':int((~v).sum()),'raw_complex_l2':float(np.linalg.norm(z[v]-r[v])/np.linalg.norm(r[v]))}
def main():
 os.environ['LAL_DATA_PATH']=str(ROM.parent)
 low,high=load_official_rom(ROM,'sub1'),load_official_rom(ROM,'sub2'); nodes=high.mf_grid_phase[(high.mf_grid_phase>=.0074)&(high.mf_grid_phase<=.014)]; i=int(torch.searchsorted(nodes,torch.tensor(.01)).item()); nodes=nodes[i-15:i+16]
 rows=[]
 for c in CASES:
  f,r=public(c); mt=sum(c[:2]); mf=f*mt*MSUN_SEC; eta=c[0]*c[1]/mt**2; ok=(mf>=float(low.mf_grid_amp[0]))&(mf<=float(high.mf_grid_amp[-1])); scale=mt*MSUN_SEC*(lal.C_SI*mt*MSUN_SEC)/(500*lal.PC_SI*1e6)*.5
  args=tuple(torch.tensor(x,dtype=torch.float64) for x in c); frac=float(eob_final_mass_fraction(*args).item()); rds=[]
  for order in ((1,2),(3,3),(3,4)):
   af=float(hofmann_final_spin(*args,fit_order=order).item()); w=float(dimensionless_qnm(2,2,0,torch.tensor(af,dtype=torch.float64))[0].item()); wl=london_qnm(torch.tensor(af,dtype=torch.float64),2,2); wl=float(complex(wl).real); rds.append({'fit_order':order,'final_mass_fraction':frac,'final_spin':af,'qnm_Mf':w/(2*PI*frac),'london_qnm_Mf':wl/(2*PI*frac)})
  for name,rd in [('public',public_qnm(c))]+[(str(x['fit_order']),x['qnm_Mf']) for x in rds]:
   try:
    a,p=evaluate_stitched_frequency(low,high,torch.as_tensor(mf[ok],dtype=torch.float64),eta,c[2],c[3],ringdown_frequency=rd,fit_nodes=nodes,reference_mf=20*mt*MSUN_SEC); z=torch.polar(a,p).detach().numpy()*scale; mm=metric(z,r[ok])
   except ValueError as exc: mm={'status':'unsupported_domain','error':str(exc)}
   rows.append({'case':c,'qnm_source':name,'qnm_Mf':rd,'metric':mm})
  rows[-1]['published_remnant_qnm_candidates']=rds
 out={'method':'independent eob remnant fits plus clean Kerr QNM versus public SimIMREOBGenerateQNMFreqV2; diagnostic only','qnm_units':'dimensionless Kerr M_final omega converted to total-mass Mf as omega/(2*pi*(M_final/M_total))','rom_sha256':h(ROM),'source_hashes':{str(x):h(x) for x in [Path(__file__),Path('src/torchwave/approximants/eob/remnant.py'),Path('src/torchwave/approximants/kerr_qnm_generated.py')]},'versions':{'python':sys.version,'torch':torch.__version__,'lal':getattr(lal,'__version__','unknown'),'lalsimulation':getattr(ls,'__version__','unknown')},'rows':rows}
 Path('docs/diagnose_rom_qnm_dependency.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
