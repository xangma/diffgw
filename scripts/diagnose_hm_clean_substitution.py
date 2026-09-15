"""Process-local Eq. (11)-(12) HM amplitude substitution experiment.

Uses only the documented binary polarization API for comparison.  The native
``h_hat_multipole`` callable is monkeypatched in memory and restored in a
``finally`` block; no production file is modified.
"""
import hashlib, json, pathlib, inspect
import numpy as np
import torch
import lal, lalsimulation as lalsim
from torchwave.approximants import imrphenom_hm as native
from torchwave.approximants.imrphenom_hm_clean import pn_hhat_lm
from torchwave.approximants.imrphenom_hm_clean import map_amplitude_frequency
from torchwave.waveforms import get_fd_waveform

MODES = ((2,2),(2,1),(3,3),(3,2),(4,4),(4,3))
CASES = ({"id":"IMRPhenomHM_diagnostic","mass1":40.,"mass2":15.,"chi1":.3,"chi2":-.2,"f_ref":20.,"distance":500.,"inclination":.8,"phic":.2,"flow":20.,"fhigh":512.,"df":.25},
         {"id":"IMRPhenomHM_holdout_5","mass1":55.,"mass2":11.,"chi1":-.4,"chi2":.2,"f_ref":27.,"distance":680.,"inclination":1.1,"phic":.71,"flow":20.,"fhigh":512.,"df":.25})

def _pn_truncations(v, eta, delta, chi_s, chi_a):
    """Mishra (2016) Eq. (12b,d) truncations, retained as diagnostics."""
    c21 = -(np.sqrt(2.0) / 3.0)
    lead21 = c21 * delta * v
    v2_spin21 = -c21 * 1.5 * (chi_a + delta * chi_s) * v**2
    v3_nonspin21 = c21 * delta * (335.0/672.0 + 117.0*eta/56.0) * v**3
    c32 = -(1.0/3.0) * np.sqrt(5.0/7.0)
    lead32 = c32 * (1.0 - 3.0*eta) * v**2
    spin32 = c32 * (4.0*eta*chi_s) * v**3
    return {'21_leading': lead21, '21_leading_plus_V2_spin': lead21 + v2_spin21,
            '21_full_through_V3': lead21 + v2_spin21 + v3_nonspin21,
            '32_leading': lead32, '32_V3_spin': spin32,
            '32_full_through_V3': lead32 + spin32}

def binary(case, active):
    ma=lalsim.SimInspiralCreateModeArray()
    for l,m in active: lalsim.SimInspiralModeArrayActivateMode(ma,l,m)
    d=lal.CreateDict(); lalsim.SimInspiralWaveformParamsInsertModeArray(d,ma)
    return lalsim.SimInspiralChooseFDWaveform(case['mass1']*lal.MSUN_SI,case['mass2']*lal.MSUN_SI,0,0,case['chi1'],0,0,case['chi2'],case['distance']*1e6*lal.PC_SI,case['inclination'],case['phic'],0,0,0,case['df'],case['flow'],case['fhigh'],case['f_ref'],d,lalsim.GetApproximantFromString('IMRPhenomHM'))

def one(case, active):
    # Construct the same full 0..fhigh grid as parity_harness; return only
    # the declared comparison band after waveform generation.
    f=torch.arange(int(round(case['fhigh']/case['df']))+1,dtype=torch.float64)*case['df']
    h,c=get_fd_waveform(approximant='IMRPhenomHM',mass1=case['mass1'],mass2=case['mass2'],chi1=case['chi1'],chi2=case['chi2'],distance=case['distance'],inclination=case['inclination'],phic=case['phic'],tc=0.,f_ref=case['f_ref'],f_lower=case['flow'],f_final=case['fhigh'],sample_frequencies=f,modes=active)
    bh,bc=binary(case,active); k=int(round(case['flow']/case['df'])); n=len(bh.data.data)-k
    return h.detach().numpy()[k:k+n],c.detach().numpy()[k:k+n],bh.data.data[k:],bc.data.data[k:]

def metric(x,y):
    return {'raw_l2':float(np.linalg.norm(x-y)/np.linalg.norm(y)), 'magnitude_l2':float(np.linalg.norm(np.abs(x)-np.abs(y))/np.linalg.norm(np.abs(y)))}

def run():
    old=native.h_hat_multipole
    native_path=pathlib.Path(native.__file__)
    report={'signature':str(inspect.signature(old)),
            'beta_convention':'London Eq. (4): [Hhat_lm(fA)/Hhat_22(fA)] [Hhat_lm(f)/Hhat_lm(2f/m)], with each Hhat_lm(x)=Eq.(12) factor times Mishra Eq.(11) V_m(x)^(-7/2), V_m(x)=(2*pi*M*x/m)^(1/3); Hhat_22 uses m=2.',
            'native_source_sha256_before':hashlib.sha256(native_path.read_bytes()).hexdigest(),
            'equation_provenance': {'mishra_eq11':'arXiv:1601.05588, Sec. III, Eq. (11), common SPA factor V_m^(-7/2)', 'mishra_eq12':'arXiv:1601.05588, Sec. III, Eqs. (12b),(12d)', 'london_eq4':'arXiv:1708.00404, Eq. (4)', 'london_alternative':'arXiv:1708.00404, paragraph immediately after Eq. (4)', 'truncation_note':'London 1.5PN-in-f wording is recorded alongside explicit Mishra through-V3 alternatives; no truncation is selected by residual.'},
            'cases':[],'versions':{'lal':lal.__version__,'lalsimulation':lalsim.__version__}}
    try:
        for case in CASES:
            rec={'id':case['id'],'modes':{},'ratios':{}}
            rec['inputs']=dict(case)
            eta=(case['mass1']*case['mass2'])/(case['mass1']+case['mass2'])**2
            delta=(case['mass1']-case['mass2'])/(case['mass1']+case['mass2'])
            cs=(case['chi1']+case['chi2'])/2.0
            ca=(case['chi1']-case['chi2'])/2.0
            for lm in MODES:
                l,m=lm; v=torch.tensor([.02,.05,.1],dtype=torch.float64); d=torch.tensor((case['mass1']-case['mass2'])/(case['mass1']+case['mass2']),dtype=torch.float64); eta=torch.tensor(case['mass1']*case['mass2']/(case['mass1']+case['mass2'])**2,dtype=torch.float64); cs=torch.tensor((case['chi1']+case['chi2'])/2,dtype=torch.float64); ca=torch.tensor((case['chi1']-case['chi2'])/2,dtype=torch.float64)
                n=old(l,m,v,d,eta,cs,ca); q=pn_hhat_lm(lm,v,eta=eta,delta=d,chi_s=cs,chi_a=ca)
                # Eq. (4) and London's explicitly stated alternative after
                # Eq. (4), evaluated with the same clean Eq. (11)-(12) factors.
                M=(case['mass1']+case['mass2'])*lal.MTSUN_SI
                fgeom=v**3/(2*np.pi*M/m); fa=2*fgeom/m
                vmfa=(2*np.pi*M*fa/m)**(1/3); vm2=(2*np.pi*M*(2*fgeom/m)/m)**(1/3)
                hmfa=pn_hhat_lm(lm,vmfa,eta=eta,delta=d,chi_s=cs,chi_a=ca)*vmfa**(-3.5); hmf=pn_hhat_lm(lm,v,eta=eta,delta=d,chi_s=cs,chi_a=ca)*v**(-3.5); hm2=pn_hhat_lm(lm,vm2,eta=eta,delta=d,chi_s=cs,chi_a=ca)*vm2**(-3.5)
                h22fa=-((2*np.pi*M*fa/2)**(1/3))**(-3.5)
                eq4=(hmfa/h22fa)*(hmf/hm2); alt=hmf/h22fa
                item={'v':v.tolist(),'native_over_clean':(n/q).tolist(),'eq4_beta':eq4.tolist(),'alternative_beta':alt.tolist(),'alternative_over_eq4':(alt/eq4).tolist()}
                if (l,m) in ((2,1),(3,2)):
                    dec=_pn_truncations(v.numpy(),float(eta),float(d),float(cs),float(ca))
                    item['clean_pn_truncations']={k:x.tolist() for k,x in dec.items() if k.startswith(f'{l}{m}_')}
                rec['ratios'][f'{l}{m}']=item
            before={}; single_arrays={}
            for active_name,active in [('22',((2,2),)),*[(f'{l}{m}',((l,m),)) for l,m in MODES[1:]],('all',MODES)]:
                h,c,bh,bc=one(case,active); before[active_name]={'hp':metric(h,bh),'hc':metric(c,bc)}
                single_arrays[active_name]=(h,c,bh,bc)
                if active_name != 'all':
                    # Direct black-box/native-to-public diagnostics.  These
                    # are reported by frequency band and are not fitted.
                    freq=np.arange(int(round(case['flow']/case['df'])),int(round(case['fhigh']/case['df']))+1)*case['df']
                    bands=[(20.,40.),(40.,80.)]
                    l,m=active[0]
                    model=native.IMRPhenomHM()
                    frlm,_=model.mode_qnm_frequencies(l,m,torch.tensor(case['mass1']),torch.tensor(case['mass2']),torch.tensor(case['chi1']),torch.tensor(case['chi2']))
                    fr22,_=model.mode_qnm_frequencies(2,2,torch.tensor(case['mass1']),torch.tensor(case['mass2']),torch.tensor(case['chi1']),torch.tensor(case['chi2']))
                    msec=(case['mass1']+case['mass2'])*lal.MTSUN_SI
                    break_hz={'f0_amplitude':float(.018*float(frlm)/float(fr22)/msec), 'f_lm_rd':float(frlm)/msec, 'f_22_rd':float(fr22)/msec}
                    bd={}
                    for lo,hi in bands:
                        mask=(freq>=lo)&(freq<hi)
                        z=h[mask]/bh[mask]
                        bd[f'{lo:g}-{hi:g}Hz']={'amplitude_ratio_median':float(np.median(np.abs(z))), 'phase_residual_circular_rms_rad':float(np.sqrt(np.mean(np.angle(z)**2)))}
                    rec['modes'].setdefault(active_name,{})['native_to_binary_by_band']=bd
                    sample_f=np.array([20.0,30.0,39.75])
                    sample_i=np.rint(sample_f/case['df']).astype(int)-int(round(case['flow']/case['df']))
                    sample_v=(2*np.pi*msec*sample_f/m)**(1.0/3.0)
                    dec=_pn_truncations(sample_v, float(eta), float(delta), float(cs), float(ca))
                    rec['modes'][active_name]['low_frequency_samples']={'frequency_hz':sample_f.tolist(),'native_over_binary_amplitude':np.abs(h[sample_i]/bh[sample_i]).tolist(),'clean_pn_truncations':{key:val.tolist() for key,val in dec.items() if key.startswith(active_name+'_')}}
                    rec['modes'][active_name]['geometric_break_frequencies_hz']=break_hz
            # Algebraic reassembly of the explicitly stated alternative beta:
            # beta_alt/beta_Eq4 = Hhat_lm(2f/m)/Hhat_lm(f22^A).
            Msec=(case['mass1']+case['mass2'])*lal.MTSUN_SI; eta=(case['mass1']*case['mass2'])/(case['mass1']+case['mass2'])**2; delta=(case['mass1']-case['mass2'])/(case['mass1']+case['mass2']); cs=(case['chi1']+case['chi2'])/2; ca=(case['chi1']-case['chi2'])/2
            freq=np.arange(int(round(case['flow']/case['df'])),int(round(case['fhigh']/case['df']))+1)*case['df']; altered={}; sumhp=np.zeros_like(freq,dtype=complex); sumhc=np.zeros_like(freq,dtype=complex); bhall,bcall=binary(case,MODES); k=int(round(case['flow']/case['df']))
            for l,m in MODES:
                name=f'{l}{m}'; h,c,bh,bc=single_arrays[name]; model=native.IMRPhenomHM(); frlm,_=model.mode_qnm_frequencies(l,m,torch.tensor(case['mass1'],dtype=torch.float64),torch.tensor(case['mass2'],dtype=torch.float64),torch.tensor(case['chi1'],dtype=torch.float64),torch.tensor(case['chi2'],dtype=torch.float64)); fr22,_=model.mode_qnm_frequencies(2,2,torch.tensor(case['mass1'],dtype=torch.float64),torch.tensor(case['mass2'],dtype=torch.float64),torch.tensor(case['chi1'],dtype=torch.float64),torch.tensor(case['chi2'],dtype=torch.float64)); f0=.018*float(frlm)/float(fr22); fa=map_amplitude_frequency(torch.tensor(freq*Msec,dtype=torch.float64),f0,float(frlm),float(fr22),m).numpy()/Msec
                vf=(2*np.pi*Msec*freq/m)**(1/3); v2=(2*np.pi*Msec*(2*freq/m)/m)**(1/3); vfa=(2*np.pi*Msec*fa/m)**(1/3); hf2=pn_hhat_lm((l,m),torch.tensor(v2),eta=eta,delta=delta,chi_s=cs,chi_a=ca).numpy()*v2**-3.5; hfa=pn_hhat_lm((l,m),torch.tensor(vfa),eta=eta,delta=delta,chi_s=cs,chi_a=ca).numpy()*vfa**-3.5; ratio=np.real(hf2/hfa); ah=h*ratio; ac=c*ratio; sumhp+=ah; sumhc+=ac; altered[name]={'hp':metric(ah,bh),'hc':metric(ac,bc),'ratio_alt_over_eq4_by_band':{}}
                for lo,hi in ((20.,40.),(40.,80.),(80.,160.)):
                    mask=(freq>=lo)&(freq<hi); altered[name]['ratio_alt_over_eq4_by_band'][f'{lo:g}-{hi:g}Hz']=float(np.median(ratio[mask]))
            altered['all']={'hp':metric(sumhp,bhall.data.data[k:]),'hc':metric(sumhc,bcall.data.data[k:])}
            native_sum_hp=sum(single_arrays[n][0] for n in ('22','21','33','32','44','43')); native_sum_hc=sum(single_arrays[n][1] for n in ('22','21','33','32','44','43'))
            altered['native_sum_single_vs_all']={'hp':metric(native_sum_hp,single_arrays['all'][0]),'hc':metric(native_sum_hc,single_arrays['all'][1])}
            rec['alternative_reassembly']={'scope':'legacy native-phase amplitude reassembly diagnostic; not an equation-level Eq.4 baseline','native_all_control':before['all'],'alternative_all':altered['all'],'modes':altered}
            # Direct London Eq. (3)-(4) reconstruction: obtain only the
            # black-box 22 carrier amplitude at f22A; both beta choices are
            # then computed wholly from the clean Eq. (11)-(12) factors.
            reconstructed = {}; eq4_hp=np.zeros_like(freq,dtype=complex); eq4_hc=np.zeros_like(freq,dtype=complex); alt_hp=np.zeros_like(freq,dtype=complex); alt_hc=np.zeros_like(freq,dtype=complex)
            model=native.IMRPhenomHM(); tf=torch.tensor(freq,dtype=torch.float64)
            for l,m in MODES:
                name=f'{l}{m}'; h,c,bh,bc=single_arrays[name]; frlm,_=model.mode_qnm_frequencies(l,m,torch.tensor(case['mass1'],dtype=torch.float64),torch.tensor(case['mass2'],dtype=torch.float64),torch.tensor(case['chi1'],dtype=torch.float64),torch.tensor(case['chi2'],dtype=torch.float64)); fr22,_=model.mode_qnm_frequencies(2,2,torch.tensor(case['mass1'],dtype=torch.float64),torch.tensor(case['mass2'],dtype=torch.float64),torch.tensor(case['chi1'],dtype=torch.float64),torch.tensor(case['chi2'],dtype=torch.float64)); f0=.018*float(frlm)/float(fr22); fa=map_amplitude_frequency(tf*Msec,f0,float(frlm),float(fr22),m).numpy()/Msec
                _,_,modes22=model(sample_frequencies=torch.tensor(fa,dtype=torch.float64),mass1=case['mass1'],mass2=case['mass2'],chi1=case['chi1'],chi2=case['chi2'],distance=case['distance'],inclination=case['inclination'],phic=case['phic'],tc=0.,f_ref=case['f_ref'],return_unprojected_modes=True)
                v=(2*np.pi*Msec*freq/m)**(1/3); v2=(2*np.pi*Msec*(2*freq/m)/m)**(1/3); vfa=(2*np.pi*Msec*fa/m)**(1/3)
                def hh(x): return pn_hhat_lm((l,m),torch.tensor(x,dtype=torch.float64),eta=eta,delta=delta,chi_s=cs,chi_a=ca).numpy()*x**-3.5
                A22=np.abs(modes22[(2,2)].detach().numpy()); qf=hh(v); q2=hh(v2); qa=hh(vfa); eq4b=np.abs((qa/(-((2*np.pi*Msec*fa/2)**(1/3))**-3.5))*(qf/q2)); altb=np.abs(qf/(-((2*np.pi*Msec*fa/2)**(1/3))**-3.5)); pred4=A22*eq4b; preda=A22*altb; native_mode=np.abs(native.IMRPhenomHM()(sample_frequencies=tf,mass1=case['mass1'],mass2=case['mass2'],chi1=case['chi1'],chi2=case['chi2'],distance=case['distance'],inclination=case['inclination'],phic=case['phic'],tc=0.,f_ref=case['f_ref'],return_unprojected_modes=True)[2][(l,m)].detach().numpy()); eq4h=h*(pred4/native_mode); eq4c=c*(pred4/native_mode); alth=h*(preda/native_mode); altc=c*(preda/native_mode); eq4_hp+=eq4h; eq4_hc+=eq4c; alt_hp+=alth; alt_hc+=altc
                low=(freq>=20.)&(freq<40.)
                reconstructed[name]={'native_unprojected_amplitude_l2':float(np.linalg.norm(native_mode-A22*eq4b)/np.linalg.norm(native_mode)),'eq4_magnitude_low_frequency_relative_l2':float(np.linalg.norm(native_mode[low]-pred4[low])/np.linalg.norm(native_mode[low])),'eq4_vs_binary_low_frequency':{'hp':metric(eq4h[low],bh[low]),'hc':metric(eq4c[low],bc[low])},'eq4_physical_hp':metric(eq4h,bh),'eq4_physical_hc':metric(eq4c,bc),'alternative_physical_hp':metric(alth,bh),'alternative_physical_hc':metric(altc,bc)}
                if name=='22': reconstructed[name]['eq4_reference_A22_identity_max_abs']=float(np.max(np.abs(eq4b-1.0)))
            reconstructed['all']={'eq4_hp':metric(eq4_hp,bhall.data.data[k:]),'eq4_hc':metric(eq4_hc,bcall.data.data[k:]),'alternative_hp':metric(alt_hp,bhall.data.data[k:]),'alternative_hc':metric(alt_hc,bcall.data.data[k:])}
            rec['direct_equation_reconstruction']={'scope':'London Eq. (3)-(4) with black-box 22 carrier amplitude; no fitted normalization','modes':reconstructed}
            patch_count={'calls':0}
            def replacement(l,m,v_m,delta,eta,chi_s,chi_a): patch_count['calls']+=1; return pn_hhat_lm((l,m),v_m,eta=eta,delta=delta,chi_s=chi_s,chi_a=chi_a)
            native.h_hat_multipole=replacement
            after={}
            try:
                for active_name,active in [('22',((2,2),)),*[(f'{l}{m}',((l,m),)) for l,m in MODES[1:]],('all',MODES)]:
                    h,c,bh,bc=one(case,active); after[active_name]={'hp':metric(h,bh),'hc':metric(c,bc)}
            finally: native.h_hat_multipole=old
            rec['before']=before; rec['after']=after; rec['patch_count']=patch_count; report['cases'].append(rec)
    finally: native.h_hat_multipole=old
    p=pathlib.Path(__file__); report['script_sha256']=hashlib.sha256(p.read_bytes()).hexdigest(); report['helper_sha256']=hashlib.sha256((p.parent.parent/'src/torchwave/approximants/imrphenom_hm_clean.py').read_bytes()).hexdigest(); report['native_source_sha256_after']=hashlib.sha256(native_path.read_bytes()).hexdigest(); pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__': run()
