"""Diagnostic of the explicit constant-part term in the Appendix-F zeta correction."""
import json
from unittest.mock import patch
import torch
import torchwave.approximants.precession_msa_angles_clean as a
from diagnose_msa_oscillatory_factor_clean import CASES, F, public_reference

def run(m1,m2,s1,s2):
    ref=public_reference(m1,m2,s1,s2)
    original=a.oscillatory_correction
    def literal_zeta(psi,J,psidot,aa,c0,c2,c4,d0,d2,d4,at,bt):
        phi,zeta=original(psi,J,psidot,aa,c0,c2,c4,d0,d2,d4,at,bt)
        return phi, zeta-J/psidot*(aa+c4/d4)*bt*(-torch.sin(2*psi)/4)
    with patch.object(a,"oscillatory_correction",literal_zeta):
        out=a.assemble_precession_msa_angles(torch.tensor(F),m1,m2,s1,s2,f_ref=20.,order=5,beta_pn=True,beta_pn_version=102,beta_compat_102=False)
    ae=[float(x-y) for x,y in zip(out['alpha'],ref[0])]; ge=[float(x-y) for x,y in zip(out['gamma'],ref[1])]
    aa=[x-ae[0] for x in ae]; gg=[x-ge[0] for x in ge]
    return {'parameters':{'mass1_msun':m1,'mass2_msun':m2,'chi1':list(s1),'chi2':list(s2),'f_ref_hz':20.,'inclination':.7,'mprime':2,'expansion_order':5,'prec_version':223},'frequencies_hz':F,'public_arrays':{'alpha':ref[0],'gamma':ref[1],'cos_beta':ref[2]},'variant':{'formula':'literal Eq.(zeta0), explicit constant term omitted','alpha_raw_error':ae,'gamma_raw_error':ge,'alpha_anchored_error':aa,'gamma_anchored_error':gg,'alpha_anchored_maxabs':max(map(abs,aa)),'gamma_anchored_maxabs':max(map(abs,gg)),'PNcosbeta_abs_error':[abs(float(x-y)) for x,y in zip(out['cos_beta_pn'],ref[2])]}}

if __name__=='__main__':
 out={'provenance':'Clean-room diagnostic of the paper-displayed zeta0 expression; runtime unchanged.','public_callable':'SimIMRPhenomXPMSAAngles','cases':[run(*c) for c in CASES]}
 with open('docs/reviews/2026-09-13-parity-review/msa-zeta-constant-term-reference.json','w') as h: json.dump(out,h,indent=2); h.write('\n')
 print(json.dumps(out,indent=2))
