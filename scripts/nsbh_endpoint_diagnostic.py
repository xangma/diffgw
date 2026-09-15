import json, math, numpy as np, torch, lal, lalsimulation as ls
from torchwave import get_fd_waveform

def call(m1,m2,c1,l2,flow,fmax,df):
 d=lal.CreateDict(); ls.SimInspiralWaveformParamsInsertTidalLambda1(d,0.); ls.SimInspiralWaveformParamsInsertTidalLambda2(d,l2)
 args=[m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,c1,0.,0.,0.,100e6*lal.PC_SI,.5,.3,0.,0.,0.,df,flow,fmax,flow,d,ls.IMRPhenomNSBH]
 hp,_=ls.SimInspiralChooseFDWaveform(*args); a=np.asarray(hp.data.data,dtype=np.complex128); f=np.arange(len(a))*df
 sel=(f>=flow)&(f<fmax)&(abs(a)>0)
 tw,_=get_fd_waveform('IMRPhenomNSBH',mass1=m1,mass2=m2,chi1=c1,chi2=0.,lambda2=l2,f_ref=flow,distance=100.,inclination=.5,phic=.3,sample_frequencies=torch.tensor(f,dtype=torch.float64),f_lower=flow,f_final=fmax,dtype=torch.float64)
 b=tw.detach().numpy(); common=sel & (abs(b)>0)
 # phase residual, with 2pi unwrap; fit affine over common in-band support
 ph=np.unwrap(np.angle(b[common]/a[common])); ff=f[common]
 X=np.c_[np.ones(len(ff)),ff]; ab=np.linalg.lstsq(X,ph,rcond=None)[0]; res=ph-X@ab
 return dict(df=df,fmax=fmax,n=len(a),last_frequency=float(f[-1]),nonzero_first=float(f[sel][0]),nonzero_last=float(f[sel][-1]),common_n=int(common.sum()),phase_at_flow=float(ph[0]),phase_at_last=float(ph[-1]),fit_intercept=float(ab[0]),fit_slope=float(ab[1]),res_rms=float(np.sqrt(np.mean(res*res))),res_max=float(np.max(abs(res))))

out=[]
for case in [(5.,1.4,.2,400.,30.),(7.,1.4,.4,2600.,40.)]:
 for df in [.25,.5,1.]:
  for fmax in [512.,1024.,1536.,2048.]: out.append(dict(case=case,**call(*case,fmax,df)))
print(json.dumps(out,indent=2))
