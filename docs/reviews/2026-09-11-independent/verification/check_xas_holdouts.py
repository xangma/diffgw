"""Public-output diagnostics; affine alignment is diagnostic, never acceptance/calibration."""
import argparse, json
from pathlib import Path
import numpy as np, torch, lal, lalsimulation as ls
from torchwave.constants import MSUN_SEC
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--json-out", type=Path, required=True)
args=parser.parse_args()
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
model=IMRPhenomXAS()
cases=[(35,30,.2,-.3),(30,30,0,0),(70,10,-.5,.2),(25,25,.8,.8),(30,15,.6,-.4),(18,12,-.7,-.2),(10,40,-.3,.5),(100,10,.8,-.8),(20,20,-.9,-.9),(60,5,0,0),(100,1,0,0),(100,1,.8,0),(50,5,-.8,.8),(45,20,.35,.6),(90,30,-.4,-.6),(12,11,.2,.3)]
rows=[]
for m1,m2,c1,c2 in cases:
 M=(m1+m2)*MSUN_SEC; mf=np.linspace(.003,.2,7881);f=mf/M
 seq=lal.CreateREAL8Sequence(len(f));seq.data[:]=f
 a,_=ls.SimInspiralChooseFDWaveformSequence(.4,m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,c1,0.,0.,c2,20.,500e6*lal.PC_SI,.6,lal.CreateDict(),ls.IMRPhenomXAS,seq)
 a=a.data.data
 b,_=model(f=torch.tensor(f),mass1=m1,mass2=m2,chi1=c1,chi2=c2,distance=500.,inclination=.6,phi_c=.4,f_ref=20.,window_type=None)
 b=b.detach().numpy();d=np.unwrap(np.angle(b/a));fit=np.polyfit(mf,d,1);res=d-np.polyval(fit,mf)
 row=dict(case=[m1,m2,c1,c2],amplitude_relative_max=float(np.max(abs(abs(b/a)-1))),amplitude_l2=float(np.linalg.norm(abs(b)-abs(a))/np.linalg.norm(a)),phase_after_affine_max=float(np.max(abs(res))),raw_complex_l2=float(np.linalg.norm(b-a)/np.linalg.norm(a)))
 rows.append(row);print(row,flush=True)
args.json_out.write_text(json.dumps({'note':'Phase affine diagnostics do not establish raw parity. No fitted value used in model.','rows':rows},indent=2)+'\n')
