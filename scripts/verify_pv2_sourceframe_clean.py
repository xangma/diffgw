"""Black-box public Pv2 source-frame comparison for the clean geometry."""
import copy, hashlib, json, math, platform, sys
from pathlib import Path
import torch
import lal, lalsimulation as ls
from torchwave.approximants.phenompv2_waveform_clean import _geometry, effective_chip
from parity_harness import get_canonical_catalog_cases, get_canonical_holdout_cases

def pd(a,b): return float((a-b+math.pi)%(2*math.pi)-math.pi)
def cases():
    out=[]
    for source in (get_canonical_catalog_cases(), get_canonical_holdout_cases()):
        out += [copy.deepcopy(c) for c in source if c['torch_name']=='IMRPhenomPv2']
    out += [dict(id='independent_generic',params=dict(mass1=40.,mass2=20.,spin1x=.3,spin1y=.2,spin1z=.4,spin2x=-.2,spin2y=.1,spin2z=-.2,inclination=.7,phic=0.,f_ref=20.)),
            dict(id='large_q',params=dict(mass1=60.,mass2=5.,spin1x=.6,spin1y=-.2,spin1z=.3,spin2x=-.2,spin2y=.1,spin2z=-.1,inclination=1.,phic=.4,f_ref=30.)),
            dict(id='aligned',params=dict(mass1=35.,mass2=25.,spin1x=0.,spin1y=0.,spin1z=.2,spin2x=0.,spin2y=0.,spin2z=-.1,inclination=.7,phic=.3,f_ref=20.))]
    return out
def one(c):
    p=c['params']; m1,m2=p['mass1'],p['mass2']; s=[p.get(k,0.) for k in ('spin1x','spin1y','spin1z','spin2x','spin2y','spin2z')]
    g=_geometry(*[torch.tensor(float(x),dtype=torch.float64) for x in (m1,m2,*s,p.get('inclination',0.),p.get('phic',0.),p['f_ref'])])
    pub=ls.SimIMRPhenomPCalculateModelParametersFromSourceFrame(m1*lal.MSUN_SI,m2*lal.MSUN_SI,p['f_ref'],p.get('phic',0.),p.get('inclination',0.),*s,ls.IMRPhenomPv2_V)
    chip=float(effective_chip(*[torch.tensor(float(x),dtype=torch.float64) for x in (m1,m2,s[0],s[1],s[3],s[4])]))
    return {'id':c['id'],'inputs':p,'public_callable':'SimIMRPhenomPCalculateModelParametersFromSourceFrame','public_enum':'IMRPhenomPv2_V','clean':[float(x) for x in g],'public':[float(x) for x in pub],'errors':{'chi_p':abs(chip-pub[2]),'theta_JN':abs(float(g[1])-pub[3]),'alpha0_periodic':abs(pd(float(g[0]),pub[4])),'phi_aligned_periodic':abs(pd(float(g[3]),pub[5])),'zeta_periodic':abs(pd(float(g[2]),pub[6]))}}
def main():
    out={'provenance':'MIT clean helper vs public LAL black-box; Pv2 source-frame API','python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'lal':getattr(lal,'__version__','unknown'),'lalsimulation':getattr(ls,'__version__','unknown'),'cases':[one(c) for c in cases()],'source_sha256':{}}
    for f in ('src/torchwave/approximants/phenompv2_waveform_clean.py',__file__): out['source_sha256'][f]=hashlib.sha256(Path(f).read_bytes()).hexdigest()
    dst=Path('docs/reviews/2026-09-13-parity-review/pv2-clean/sourceframe-geometry-v2.json'); dst.parent.mkdir(parents=True,exist_ok=True); dst.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
