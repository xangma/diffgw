"""Intrinsic Pv2 black-box comparison (public FrequencySequence API only)."""
import hashlib, json, math, pathlib
import numpy as np, torch, lal, lalsimulation as ls
torch.set_default_dtype(torch.float64)
from torchwave.approximants.phenompv2_waveform_clean import _L2
from torchwave.approximants.precession_nnlo_clean import raw_angles
from torchwave.approximants.precession_twisting_clean import project_twisted_positive_modes
from torchwave.approximants.imrphenomd import IMRPhenomD

CASES=[dict(m1=35.,m2=25.,z1=.2,z2=-.1,chip=.3,theta=.7,alpha=.4,phic=.3),
       dict(m1=45.,m2=15.,z1=.3,z2=-.25,chip=.6,theta=1.2,alpha=-.3,phic=.2)]
CASES += [dict(x, phic=0.) for x in CASES]

def metrics(a,b):
    return {'relative_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)),
            'max_normalized_error':float(np.max(np.abs(a-b))/max(np.max(np.abs(b)),1e-300))}

def main():
    f=np.arange(20.,513.,1.); fv=lal.CreateREAL8Vector(len(f)); fv.data[:]=f
    rows=[]
    for c in CASES:
        vals=[]
        ff=torch.tensor(f,dtype=torch.float64); m1,m2=c['m1'],c['m2']; Msec=(m1+m2)*4.925490947641267e-6
        eta=torch.tensor(m1*m2/(m1+m2)**2,dtype=torch.float64); delta=torch.tensor((m1-m2)/(m1+m2),dtype=torch.float64)
        v=(math.pi*Msec*ff).pow(1/3); vr=torch.tensor(math.pi*Msec*20,dtype=torch.float64).pow(1/3)
        assert 0.1 < float(v[0]) < 0.5
        beta=torch.atan2(torch.tensor(c['chip']*(m1/(m1+m2))**2),torch.abs(_L2(torch.tensor(m1),torch.tensor(m2),ff)+torch.tensor((m1/(m1+m2))**2*c['z1']+(m2/(m1+m2))**2*c['z2'])))
        vals=[]
        for label,chil in [('z1',c['z1']),('chieff_over_x1',((m1*c['z1']+m2*c['z2'])/(m1+m2))/(m1/(m1+m2)))]:
            aa,ee=raw_angles(v,eta,-delta,torch.tensor(m1/(m1+m2)),torch.tensor(chil),torch.tensor(c['chip'])); a0,e0=raw_angles(vr,eta,-delta,torch.tensor(m1/(m1+m2)),torch.tensor(chil),torch.tensor(c['chip']))
            def rem(a,b,za,zb):
                mf,afd,frd,fdd=IMRPhenomD.remnant_properties(a,b,za,zb); ap=torch.sqrt(afd.square()+(torch.tensor(c['chip'])*(a/(a+b)).square()).square()); ap=torch.where(afd<0,-ap,ap); wr,wd=__import__('torchwave.approximants.kerr_qnm_generated',fromlist=['signed_qnm_220']).signed_qnm_220(ap); return mf,ap,wr/(2*math.pi)/(mf/(a+b)),wd/(2*math.pi)/(mf/(a+b))
            # Intrinsic Pv2 uses D's coalescence time origin; f_ref remains
            # exclusively the precession-angle anchor.
            car,_=IMRPhenomD()(ff,m1,m2,c['z1'],c['z2'],distance=500.,inclination=0.,phic=c['phic'],f_ref=20.,phasing_convention='coalescence',remnant_provider=rem)
            p,h=project_twisted_positive_modes({(2,2):2*car/math.sqrt(5/(4*math.pi))},aa-a0+c['alpha'],-ee+e0,beta,c['theta'],0.)
            up,uc=project_twisted_positive_modes({(2,2):torch.ones_like(car)},aa-a0+c['alpha'],-ee+e0,beta,c['theta'],0.)
            # Analytic provider diagnostic: evaluate d phase / df at
            # the remnant f_RD and apply exp(+i f k).  k remains attached to
            # autograd while forming the diagnostic factor; this is never a
            # runtime calibration.
            frd_dimless=rem(torch.tensor(m1),torch.tensor(m2),torch.tensor(c['z1']),torch.tensor(c['z2']))[2]
            frd=frd_dimless/Msec
            fseed=torch.as_tensor(frd,dtype=torch.float64).clone().detach().requires_grad_(True)
            _,_,phseed=IMRPhenomD()(fseed.reshape(1),m1,m2,c['z1'],c['z2'],distance=500.,inclination=0.,phic=0.,tc=0.,f_ref=20.,phasing_convention='coalescence',return_phase=True,remnant_provider=rem)
            k=torch.autograd.grad(phseed.reshape(-1)[0],fseed,create_graph=True)[0]
            car_corr=car*torch.exp(1j*ff*k)
            p_corr,h_corr=project_twisted_positive_modes({(2,2):2*car_corr/math.sqrt(5/(4*math.pi))},aa-a0+c['alpha'],-ee+e0,beta,c['theta'],0.)
            pub=ls.SimIMRPhenomPFrequencySequence(fv,c['z1'],c['z2'],c['chip'],c['theta'],c['m1']*lal.MSUN_SI,c['m2']*lal.MSUN_SI,500e6*lal.PC_SI,c['alpha'],c['phic'],20.,ls.IMRPhenomPv2_V,ls.NoNRT_V,lal.CreateDict())
            hp=np.asarray(pub[0].data.data).copy(); hc=np.asarray(pub[1].data.data).copy()
            hc_pub=hc
            p_np=p.detach().numpy().copy(); h_np=h.detach().numpy().copy()
            p_corr_np=p_corr.detach().numpy().copy(); h_corr_np=h_corr.detach().numpy().copy()
            ratio=(p_np/h_np); pr=hp/hc
            idx=[int(x-20) for x in (20,30,50,100,200,400)]
            uu=np.asarray(up.detach().numpy()).copy(); vv=np.asarray(uc.detach().numpy()).copy(); safe=(np.abs(uu)>1e-12)&(np.abs(vv)>1e-12)
            inferred=hp/uu; inferred_hc=hc/vv
            refh=2*car.detach().numpy()/math.sqrt(5/(4*math.pi))
            phase=np.unwrap(np.angle(inferred[safe]/refh[safe])); fs=f[safe]; co=np.polyfit(fs,phase,1); resid=phase-(co[0]*fs+co[1])
            def carrier_stats(carrier):
                ii=hp/uu; jj=hc_pub/vv; rr=2*carrier.detach().numpy()/math.sqrt(5/(4*math.pi))
                ph=np.unwrap(np.angle(ii[safe]/rr[safe])); cc=np.polyfit(fs,ph,1); re=ph-(cc[0]*fs+cc[1])
                return {'magnitude_relative_l2':metrics(np.abs(ii[safe]),np.abs(rr[safe]))['relative_l2'],'unwrapped_phase_residual':ph.tolist(),'affine_diagnostic_slope':float(cc[0]),'affine_diagnostic_intercept':float(cc[1]),'max_nonlinear_residual':float(np.max(np.abs(re))),'hc_magnitude_relative_l2':metrics(np.abs(jj[safe]),np.abs(rr[safe]))['relative_l2'],'hp_hc_complex_relative_l2':float(np.linalg.norm(ii[safe]-jj[safe])/max(np.linalg.norm(ii[safe]),1e-300))}
            vals.append({'chi_l_variant':label,'frd_hz':float(frd.detach()),'dphase_df_at_frd':float(k.detach()),'ratio_metrics':metrics(ratio,pr),'hp':metrics(p_np,hp),'hc':metrics(h_np,hc_pub),'carrier':carrier_stats(car),'analytic_frd_shift':{'hp':metrics(p_corr_np,hp),'hc':metrics(h_corr_np,hc_pub),'carrier':carrier_stats(car_corr)}})
        rows.append({'parameters':c,'variants':vals,'public_cache':{'hp_real':hp.real.tolist(),'hp_imag':hp.imag.tolist(),'hc_real':hc.real.tolist(),'hc_imag':hc.imag.tolist()}})
    out=pathlib.Path('docs/reviews/2026-09-13-parity-review/pv2-clean');out.mkdir(parents=True,exist_ok=True)
    mods=['torchwave.approximants.phenompv2_waveform_clean','torchwave.approximants.precession_nnlo_clean','torchwave.approximants.precession_twisting_clean','torchwave.approximants.imrphenomd','torchwave.approximants.kerr_qnm_generated']
    hashes={m:hashlib.sha256(pathlib.Path(__import__(m,fromlist=['__file__']).__file__).read_bytes()).hexdigest() for m in mods}
    report={'provenance':'public SimIMRPhenomPFrequencySequence black box; clean fresh geometry/twisting and opaque D carrier; no fitting','source_evidence':{'technical_note':'T1500602-v4 Eq. (17)-(19) gives the frequency-domain plus/cross twist and shows the carrier phase enters as exp[-i(phi_D+2 epsilon)]; Sec. 4 Eq. (40) defines alpha(f_ref), and the text immediately after Eq. (40) explicitly chooses epsilon(f_ref)=0 because it is degenerate with the PhenomD coalescence phase. These statements support the phase/polarization controls; the analytic dphi_D/df at f_RD multiplier remains a measured compatibility hypothesis, not a published prescription.','next_experiment':'Compare the public carrier phase and timestamp under coalescence versus f_ref phasing at identical frequencies, then test only the published constant epsilon gauge and documented time-origin controls.'},'variants':['chi_l=spin1z (technical note)','chi_l=chieff/x1 (legacy discrete comparison)'],'frequency_grid':'20..512 Hz, df=1 Hz','cases':rows,'versions':{'lal':lal.__version__,'lalsimulation':ls.__version__,'torch':torch.__version__},'component_sha256':hashes,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    (out/'intrinsic-diagnostic.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
