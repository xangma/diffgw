"""Black-box waveform comparison for the clean London HM assembly candidate."""
import hashlib, json, pathlib, sys
import numpy as np, torch, lal, lalsimulation
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.hm_assembly_clean import (
    assemble_mode, project_positive_modes, MODES,
    COMPATIBILITY_AMPLITUDE_JOIN, COMPATIBILITY_PHASE_JOIN,
)
from torchwave.approximants.kerr_qnm_london_published import dimensionless_qnm

MTSUN = 4.925490947641267e-6
CASES = ((40.,15.,.3,-.2,500.,.8,.2,20.), (55.,11.,-.4,.2,680.,1.1,.71,27.),
         (32.,18.,-.6,.45,430.,.8,.0,25.), (70.,8.,.1,-.5,900.,.9,.4,30.),
         (48.,13.,.72,-.31,610.,.7,.3,22.), (63.,19.,-.27,.58,740.,1.0,.5,31.))
FREQ = np.unique(np.r_[np.linspace(20.,512.,121), [20.,50.,100.,200.]])

def _native_modes(f, c):
    m1,m2,s1,s2,_,_,_,fref=c; v=lal.CreateREAL8Vector(len(f)); v.data[:]=f
    phic=c[6]
    # Same public callable contract as the independently verified helper:
    # vector, SI masses/spins, phiRef, deltaF, f_ref, dictionary.
    df=float(f[1]-f[0]) if len(f)>1 else 0.
    h=lalsimulation.SimIMRPhenomHMGethlmModes(v,m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,s1,0.,0.,s2,phic,df,fref,lal.CreateDict())
    out={}; n=h
    while n: out[(int(n.l),int(n.m))]=np.asarray(n.mode.data.data,dtype=np.complex128); n=n.next
    return out

def _qnm(mode, af, mt, mf):
    z=dimensionless_qnm(af, *mode); return float(z.real/(2*np.pi)*mt/mf), float(abs(z.imag)/(2*np.pi)*mt/mf)

def _row(c):
    m1,m2,s1,s2,dist,iota,phic,fref=c; mt=m1+m2; M=mt*MTSUN; eta=m1*m2/mt**2; delta=(m1-m2)/mt; cs=(s1+s2)/2; ca=(s1-s2)/2
    d=IMRPhenomD(); args=[torch.tensor(q,dtype=torch.float64) for q in (m1,m2,s1,s2)]
    mf,af,frd22_d,fdamp22_d=d.remnant_properties(*args); mf=float(mf); af=float(af)
    frd22_d=float(frd22_d); fdamp22_d=float(fdamp22_d)
    chi_pn=(m1*s1+m2*s2)/mt-38*eta/113*(s1+s2)
    a4=float(d.eval_table_v('alpha4',torch.tensor(eta,dtype=torch.float64),torch.tensor(chi_pn,dtype=torch.float64)))
    a5=float(d.eval_table_v('alpha5',torch.tensor(eta,dtype=torch.float64),torch.tensor(chi_pn,dtype=torch.float64)))
    slope=float(d.compute_slope_peak(*args))
    q={mode:_qnm(mode,af,mt,mf) for mode in MODES}
    scale=mt*lal.MTSUN_SI*(mt*lal.MRSUN_SI)/(dist*1e6*lal.PC_SI)
    raw=_native_modes(FREQ,c)
    def carrier(x):
        f=x/M; t=torch.as_tensor(f,dtype=torch.float64)
        return d(t,m1,m2,s1,s2,distance=dist,inclination=0.,phic=0.,tc=0.,f_ref=fref,return_phase=True)[2]
    def amp22(x):
        f=x/M
        hp,hc=d(f,m1,m2,s1,s2,distance=dist,inclination=0.,phic=0.,tc=0.,f_ref=None)
        y=complex(lal.SpinWeightedSphericalHarmonic(0.,0.,-2,2,2))
        return torch.abs((hp+1j*hc)/y)
    # project_positive_modes supplies (phi, theta); LAL takes (theta, phi).
    harm=lambda phi,theta,s,l,m: lal.SpinWeightedSphericalHarmonic(float(theta),float(phi),int(s),int(l),int(m))
    target_modes={k:torch.tensor(v*scale,dtype=torch.complex128) for k,v in raw.items()}
    thp,thc=project_positive_modes(target_modes,iota,harm)
    variants={}
    for label,base in [('london22_subtraction',(None,None)),('D22_subtraction',(frd22_d,fdamp22_d))]:
        pred={}
        for mode in MODES:
            gauge=((mode[1]/2)-1.0)*slope*M*fref
            z=assemble_mode(torch.tensor(FREQ*M,dtype=torch.float64),mode,eta=eta,delta=delta,chi_s=cs,chi_a=ca,carrier_positive=carrier,amplitude_22=amp22,amp_qnm_22=q[(2,2)][0],amp_qnm_lm=q[mode][0],phase_qnm_22=q[(2,2)][0],phase_qnm_lm=q[mode][0],phase_damp_22=q[(2,2)][1],phase_damp_lm=q[mode][1],alpha4=a4,alpha5=a5,chi_pn=chi_pn,phic=phic,reference_gauge=gauge,use_eq14=True,base_fr22=base[0],base_fd22=base[1],amplitude_join=COMPATIBILITY_AMPLITUDE_JOIN,phase_join=COMPATIBILITY_PHASE_JOIN)
            pred[mode]=z.detach().numpy()
        rows={}
        for mode in MODES:
            target=raw[mode]*scale; err=pred[mode]-target
            dphi=np.angle(pred[mode]*np.conj(target))
            da=np.abs(pred[mode])-np.abs(target)
            rows[str(mode)]={'relative_l2':float(np.linalg.norm(err)/np.linalg.norm(target)),'max_relative':float(np.max(np.abs(err)/np.maximum(np.abs(target),1e-300))), 'phase_rms':float(np.sqrt(np.mean(dphi*dphi))), 'phase_max':float(np.max(np.abs(dphi))), 'amplitude_relative_l2':float(np.linalg.norm(da)/np.linalg.norm(target)), 'amplitude_max_relative':float(np.max(np.abs(da)/np.maximum(np.abs(target),1e-300)))}
        pred_modes={k:torch.tensor(v,dtype=torch.complex128) for k,v in pred.items()}
        php,phc=project_positive_modes(pred_modes,iota,harm)
        variants[label]={'per_mode':rows,'polarization':{'hp_relative_l2':float(torch.linalg.vector_norm(php-thp)/torch.linalg.vector_norm(thp)),'hc_relative_l2':float(torch.linalg.vector_norm(phc-thc)/torch.linalg.vector_norm(thc))},'base_fr22':base[0],'base_fd22':base[1]}
    return {'inputs':c,'qnm_london_total_M':{str(k):v for k,v in q.items()},'d_carrier_qnm_total_M':[frd22_d,fdamp22_d],'variants':variants,'sample_count':len(FREQ)}

def main():
    out={'recipe':'compatibility joins amplitude=.014/phase=.018 (London printed defaults are amplitude=.018/phase=.014); Eq14 inside positive-D carrier; 21 Arun TD-sign control; 32 leading-only; all other modes leading; London printed QNM for mapping and Eq14','qnm_distinction':'D carrier itself remains the public D callable; London QNM values are used only as higher-mode map/merger inputs.','cases':[_row(c) for c in CASES],'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'versions':{'lal':lal.__version__,'lalsimulation':lalsimulation.__version__,'torch':torch.__version__},'clean_room':'public callable/black-box APIs and published equations; diagnostic only'}
    pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__': main()
