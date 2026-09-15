"""Compare ROM submodels against public SEOBNRv4_ROM near boundaries."""
import hashlib,json,os
from pathlib import Path
import lal,lalsimulation as ls,numpy as np,torch
from torchwave.approximants.rom.bspline import load_official_rom
from torchwave.approximants.rom.glue import OfficialROMGlue
ROM=Path('/private/tmp/torchwave-oracle-data/SEOBNRv4ROM_v3.0.hdf5'); MS=4.925490947641267e-6; APP=ls.GetApproximantFromString('SEOBNRv4_ROM')
def masses(eta,total=1000.):
 q=(1-2*eta+np.sqrt(1-4*eta))/(2*eta); return total*q/(1+q),total/(1+q)
def main():
 os.environ['LAL_DATA_PATH']=str(ROM.parent); lo=load_official_rom(ROM,'sub1'); sub2=load_official_rom(ROM,'sub2'); sub3=load_official_rom(ROM,'sub3'); rows=[]
 for eta in (.024,.025,.026):
  for chi in (.994,.995,.996):
   m1,m2=masses(eta); hp,hc=ls.SimInspiralChooseFDWaveform(m1*lal.MSUN_SI,m2*lal.MSUN_SI,0,0,chi,0,0,0,100e6*lal.PC_SI,0,0,0,0,0,.1,1,20,1,None,APP)
   v=lal.CreateCOMPLEX16Vector(1); ls.SimIMREOBGenerateQNMFreqV2(v,m1,m2,[0,0,chi],[0,0,0],2,2,1,ls.GetApproximantFromString('SEOBNRv4')); rd=float(v.data[0].real*(m1+m2)*MS/(2*np.pi))
   hp,hc=ls.SimInspiralChooseFDWaveform(m1*lal.MSUN_SI,m2*lal.MSUN_SI,0,0,chi,0,0,0,100e6*lal.PC_SI,0,0,0,0,0,.1,1,20,1,None,APP)
   f=np.arange(hp.data.length,dtype=float)*hp.deltaF; ft=torch.tensor(f,dtype=torch.float64)
   def run(high): return OfficialROMGlue(lo,high,ringdown_mf=rd,f_lower_hz=1,f_final_hz=20)(ft,m1,m2,chi,0,distance_mpc=100,fref_hz=1)
   rp=hp.data.data; rc=hc.data.data
   l2=lambda x,y:float(np.linalg.norm(x-y)/np.linalg.norm(y))
   row={'eta_requested':eta,'eta_computed':m1*m2/(m1+m2)**2,'chi1':chi,'mass1':m1,'mass2':m2,'delta_f_hz':float(hp.deltaF),'frequency_start_hz':float(f[0])}
   for name in ('sub2','sub3'):
    try:
     p,c=run(sub2 if name=='sub2' else sub3); row[name+'_hp_l2']=l2(p.detach().numpy(),rp); row[name+'_hc_l2']=l2(c.detach().numpy(),rc)
    except Exception as exc: row[name+'_error']=type(exc).__name__+': '+str(exc)
   rows.append(row)
 out={'contract':'public black-box dispatch diagnostic; no published boundary claim','rom_sha256':hashlib.sha256(ROM.read_bytes()).hexdigest(),'rows':rows}; Path('scripts/diagnose_rom_submodel_boundary.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
