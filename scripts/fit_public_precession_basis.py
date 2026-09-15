"""Diagnostic basis identification for public PNAngles; no fitted values enter runtime."""
import json, lal, lalsimulation, numpy as np, torch
from torchwave.approximants.precession_nnlo_clean import coefficients,effective_chip
CASES=[(40,20,.3,.2,.4,-.2,.1,-.2),(30,10,.1,-.25,-.4,.15,.05,.2),(20,20,.2,.1,.3,-.1,.2,-.1),(50,5,-.2,.3,.1,.1,-.1,-.3)]
MSEC=float(lal.MTSUN_SI)

def pub(c,fs,ver=102):
 v=lal.CreateREAL8Vector(len(fs));v.data[:]=np.asarray(fs,dtype=np.float64)
 d=lal.CreateDict(); lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(d,int(ver))
 o=lalsimulation.SimIMRPhenomXPPNAngles(v,c[0]*lal.MSUN_SI,c[1]*lal.MSUN_SI,*c[2:],.7,40,2,d)
 return np.array(o[0].data),np.array(o[1].data)
def fit(c,which):
 fs=np.geomspace(15,500,24); vh=(np.pi*(c[0]+c[1])*MSEC*fs)**(1/3); X=np.column_stack([vh**-3,vh**-2,vh**-1,vh,np.log(vh**3),np.ones_like(vh)])
 y=pub(c,fs)[which]; co=np.linalg.lstsq(X,y,rcond=None)[0]; held=np.array([18.,27.,63.,137.,311.]); vh2=(np.pi*(c[0]+c[1])*MSEC*held)**(1/3); X2=np.column_stack([vh2**-3,vh2**-2,vh2**-1,vh2,np.log(vh2**3),np.ones_like(vh2)]); yh=pub(c,held)[which]; return co,float(np.max(np.abs(X@co-y))),float(np.max(np.abs(X2@co-yh)))

def main():
 rows=[]
 for c in CASES:
  m1,m2,*sp=c; eta=m1*m2/(m1+m2)**2; d=(m1-m2)/(m1+m2); cl=(m1*sp[2]+m2*sp[5])/m1; cp=float(effective_chip(torch.tensor(m1,dtype=torch.float64),torch.tensor(m2,dtype=torch.float64),*map(lambda x: torch.tensor(x,dtype=torch.float64),(sp[0],sp[1],sp[3],sp[4])))); aa,ee=coefficients(torch.tensor(eta,dtype=torch.float64),torch.tensor(-d,dtype=torch.float64),torch.tensor(m1/(m1+m2),dtype=torch.float64),torch.tensor(cl,dtype=torch.float64),torch.tensor(cp))
  for w,name in [(0,'alpha'),(1,'epsilon')]:
   co,res,held=fit(c,w); expected=[float(x) for x in ((aa if w==0 else -ee).tolist())]+[0.0]; rows.append({'case':[m1,m2], 'angle':name,'clean_coefficients_minus_constant':([float(x) for x in (aa if w==0 else ee)]+[0.0]),'public_fit_coefficients':[float(x) for x in co],'expected_from_paper_with_public_conventions':expected,'coefficient_error':[float(a-b) for a,b in zip(co[:5],expected[:5])],'sample_max_residual':res,'heldout_max_residual':held})
 print(json.dumps({'basis':['v^-3','v^-2','v^-1','v','log(v^3)','1'],'frequencies_hz_fit':'geomspace(15,500,24)','heldout_hz':[18,27,63,137,311],'rows':rows},indent=2))
if __name__=='__main__': main()
