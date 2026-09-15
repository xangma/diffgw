"""Whole-mode clean-room control for the London PhenomHM construction.

This uses only public LAL HM mode output as the target and as the native
amplitude control.  Eq. (4)'s amplitude *structure* is evaluated with the
black-box native amplitude ratio; no unpublished PN coefficients are copied.
The phase uses the printed London QNM fits for both map endpoints.  This is
diagnostic code and does not route production waveforms through these values.
"""
import json, pathlib, hashlib
import numpy as np
import torch, lal, lalsimulation
from torchwave.approximants.imrphenom_hm import IMRPhenomHM
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_hm_phase_clean import mapped_phase_c1, phenomd_eq14_phase_delta_c1
from torchwave.approximants.kerr_qnm_london_published import dimensionless_qnm

MTSUN = 4.925490947641267e-6
CASES=((40.,15.,.3,-.2,20.),(55.,11.,-.4,.2,27.),(32.,18.,-.6,.45,25.),(70.,8.,.1,-.5,30.))
MODES=((2,2),(2,1),(3,3),(3,2),(4,4),(4,3))

def modes(f,m1,m2,s1,s2,rf):
    v=lal.CreateREAL8Vector(len(f)); v.data[:]=np.asarray(f,float)
    h=lalsimulation.SimIMRPhenomHMGethlmModes(v,m1*lal.MSUN_SI,m2*lal.MSUN_SI,0.,0.,s1,0.,0.,s2,0.,float(f[1]-f[0]) if len(f)>1 else 0.,rf,lal.CreateDict())
    out={}; n=h
    while n:
        out[(int(n.l),int(n.m))]=torch.tensor(n.mode.data.data,dtype=torch.complex128); n=n.next
    return out

def rms(a,b):
    d=torch.angle(a*torch.conj(b)); return {"phase_rms_rad":float(torch.sqrt(torch.mean(d*d))),"relative_complex_l2":float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))}

def project(mode_map, theta):
    hp=torch.zeros_like(next(iter(mode_map.values())),dtype=torch.complex128); hc=hp.clone()
    for (ell,m),z in mode_map.items():
        yp=lal.SpinWeightedSphericalHarmonic(theta,0.,-2,ell,m); yn=lal.SpinWeightedSphericalHarmonic(theta,0.,-2,ell,-m); p=(-1)**ell
        hp += .5*z*(yp+p*np.conj(yn)); hc += -.5j*z*(yp-p*np.conj(yn))
    return hp,hc

def main():
    hm,d=IMRPhenomHM(),IMRPhenomD(); f=torch.linspace(20.,512.,257,dtype=torch.float64); out=[]
    for m1,m2,s1,s2,rf in CASES:
        M=(m1+m2)*MTSUN; args=tuple(torch.tensor(z,dtype=torch.float64) for z in (m1,m2,s1,s2)); mf,af,dfr22,dfd22=map(float,d.remnant_properties(*args)); ratio=(m1+m2)/mf
        h22=dimensionless_qnm(af,2,2); q22=(h22.real/(2*np.pi)*ratio,h22.imag/(2*np.pi)*ratio)
        hp_native,hc_native,native=hm(f,mass1=m1,mass2=m2,chi1=s1,chi2=s2,distance=500.,inclination=.8,phic=0.,tc=0.,f_ref=rf,f_lower=20.,return_unprojected_modes=True)
        slope=float(d.compute_slope_peak(*args))
        target=modes(f.numpy(),m1,m2,s1,s2,rf)
        distance_mpc=500.; S=(m1+m2)*lal.MTSUN_SI*((m1+m2)*lal.MRSUN_SI)/(distance_mpc*1e6*lal.PC_SI)
        target={k:S*v for k,v in target.items()}; rows={}; preds={}; d_preds={}
        hp0,hc0=project({k:((-1)**k[0])*v for k,v in native.items()},.8)
        assert float(torch.linalg.vector_norm(hp0-hp_native)/torch.linalg.vector_norm(hp_native)) < 1e-12
        assert float(torch.linalg.vector_norm(hc0-hc_native)/torch.linalg.vector_norm(hc_native)) < 1e-12
        for ell,m in MODES:
            q=dimensionless_qnm(af,ell,m); fl,fd=q.real/(2*np.pi)*ratio,q.imag/(2*np.pi)*ratio; f0=.018*fl/q22[0]
            eta=m1*m2/(m1+m2)**2; chip=(m1*s1+m2*s2)/(m1+m2)-38*eta/113*(s1+s2)
            # D carrier is retained; London endpoints control only map/Eq14.
            def c0(x): return d(x/M,m1,m2,s1,s2,phic=0.,tc=0.,f_ref=rf,return_phase=True)[2]
            a4=float(d.eval_table_v('alpha4',torch.tensor(eta,dtype=torch.float64),torch.tensor(chip,dtype=torch.float64))); a5=float(d.eval_table_v('alpha5',torch.tensor(eta,dtype=torch.float64),torch.tensor(chip,dtype=torch.float64)))
            def cp(x): return c0(x)+phenomd_eq14_phase_delta_c1(x,eta,chip,q22[0],fl,q22[1],fd,a4,a5,qnm_ratio=q22[0]/fl)
            def cp_dcenter(x): return c0(x)+phenomd_eq14_phase_delta_c1(x,eta,chip,dfr22,fl,dfd22,fd,a4,a5,qnm_ratio=q22[0]/fl)
            # D is reference/peak-slope gauged; restore the analytic London
            # mode gauge for comparison at the requested f_ref.
            gauge=(m/2-1)*slope*M*rf
            phase=-mapped_phase_c1(f*M,f0,fl,q22[0],m,cp,ell=ell)+gauge
            phase_dcenter=-mapped_phase_c1(f*M,f0,fl,q22[0],m,cp_dcenter,ell=ell)+gauge
            # The public mode generator and HM wrapper use different overall
            # normalizations. Carry the public binary amplitude through both
            # predictors to isolate the phase/map effect without fitting a
            # scale.
            amp_native=torch.abs(native[(ell,m)])
            z_nativeamp=amp_native*torch.exp(1j*phase); z=target[(ell,m)]
            z_dcenter=amp_native*torch.exp(1j*phase_dcenter)
            rows[f'{ell}{m}']={'qnm_total_Mf':{'f_rd':fl,'f_damp':fd},'phase_map_f0':f0,
                'native_amplitude_clean_phase':rms(z_nativeamp,z),'eq4_amplitude_control':None,
                'D_center_London_ratio_lm_damping':rms(z_dcenter,z),
                'amplitude_definition':'Native wrapper amplitude; public raw modes use analytic mass-distance factor S; Eq4 PN beta omitted'}
            preds[(ell,m)]=z_nativeamp
            d_preds[(ell,m)]=z_dcenter
        # Sum positive-m spin-weighted modes; this is an independent angular
        # projection control, with the same convention applied to target/model.
        pol_target=project(target,.8); pol_model=project(preds,.8)
        dproj=project(d_preds,.8)
        rows['summed_polarization']={'baseline_hp':rms(hp_native,pol_target[0]),'baseline_hc':rms(hc_native,pol_target[1]),'clean_hp':rms(pol_model[0],pol_target[0]),'clean_hc':rms(pol_model[1],pol_target[1]),'D_center_hp':rms(dproj[0],pol_target[0]),'D_center_hc':rms(dproj[1],pol_target[1]),'baseline_reconstruction_assertion':'project(parity-adjusted native)==native hp/hc'}
        out.append({'mass1':m1,'mass2':m2,'f_ref_hz':rf,'Msec':M,'qnm_source':'London 1810.03550 Appendix A printed fits; D carrier retained','modes':rows})
    meta={'source':'https://arxiv.org/html/1708.00404 Eq.3-5, https://arxiv.org/html/1810.03550 Appendix A Eq.21-32','grid_hz':[20.,512.,257],'cases':out,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n'); print(pathlib.Path(__file__).with_suffix('.json'))
if __name__=='__main__': main()
