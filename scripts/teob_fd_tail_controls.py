#!/usr/bin/env python3
"""Hold a frozen waveform's physical endpoint fixed while refining auxiliary knots."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from teob_continuous_fd_attribution import OUT, case_for, json_value, source_hashes
from torchwave.approximants.eob.teob_ringdown_assembly_clean import assemble_teob_ringdown_22, natural_cubic_interpolate
from torchwave.approximants.eob.teob_uib2016_remnant import uib2016_nonspinning_remnant
from torchwave.approximants.eob.teob_conditioned_waveform_clean import project_teob22_generic
from torchwave.approximants.eob.td_fd_conditioning_clean import condition_td_to_fd, plan_td_support

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--case',default='bbh_reference');args=ap.parse_args()
    folder=OUT/args.case;path=folder/'continuous.npz';d=np.load(path);public=np.load(folder/'public-fd.npz')
    p,g=case_for(args.case)['params'],case_for(args.case)['grid']
    get=lambda k:torch.as_tensor(d[k],dtype=torch.float64)
    t,a,ph=[get('nqc_original_'+k) for k in ('time','corrected_amplitude','corrected_phase')]
    omega=get('nqc_original_orbital_frequency');start=get('nqc_auxiliary_time')[0]
    # Fixed physical endpoint chosen from the accepted trajectory, before scores.
    end=t[-1]+150.;nu=p['mass1']*p['mass2']/(p['mass1']+p['mass2'])**2
    mf,spin=uib2016_nonspinning_remnant(p['mass1'],p['mass2']);mf,spin=mf.double(),spin.double()
    support=plan_td_support(p['mass1'],p['mass2'],g['flow'],dtype=torch.float64)
    mask=(public['frequencies']>=g['flow'])&(public['frequencies']<g['fhigh'])
    report={'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_hashes_before':source_hashes(),'endpoint_policy':'last accepted time + 150 M; chosen independently of public samples','support_end_M':float(end),'variants':{},'notes':'NQC coefficients and accepted corrected waveform held fixed. Auxiliary series is resampled only for ringdown metadata. No fitted alignment.'}
    outputs={}; arrays={}
    with torch.no_grad():
        for spacing in (.05,.1,.2,.5):
            aux=start+torch.arange(int(torch.floor((t[-2]-start)/spacing))+1,dtype=torch.float64)*spacing
            aux_a=natural_cubic_interpolate(t,a,aux);aux_ph=natural_cubic_interpolate(t,ph,aux);aux_om=natural_cubic_interpolate(t,omega,aux)
            for fixed in (False,True):
                started=time.time();key=f'{"fixed" if fixed else "legacy"}_{spacing:g}'
                rd=assemble_teob_ringdown_22(t,a,ph,aux,aux_a,aux_ph,aux_om,nu,mf,spin,
                    output_step=get('sampling_dt_seconds')/get('total_mass_seconds'),matching_method='continuous',peak_time=get('event_peak_time'),support_end=end if fixed else None)
                hp,hc=project_teob22_generic(rd.mode,rd.mode.conj(),mass1=p['mass1'],mass2=p['mass2'],distance=p['distance'],inclination=p['inclination'],phic=p['phic'])
                fd=condition_td_to_fd(hp,hc,dt=get('sampling_dt_seconds'),delta_f=g['df'],f_max=g['fhigh'],support=support,tc=0.,return_metadata=True)
                arr=np.stack((fd.h_plus.numpy(),fd.h_cross.numpy()))
                ref=np.stack((public['h_plus_fd'],public['h_cross_fd']))
                outputs[key]=(rd.time.numpy().copy(),rd.mode.numpy().copy(),arr.copy())
                report['variants'][key]={'support_end_M':float(rd.support_end),'output_end_M':float(rd.time[-1]),'td_samples':hp.numel(),'extension_samples':int(rd.extension_samples),'raw_l2_vs_public':[float(np.linalg.norm(x[mask]-y[mask])/np.linalg.norm(y[mask])) for x,y in zip(arr,ref)],'seconds':time.time()-started}
                arrays[key+'_fd']=arr
                print(json.dumps({'case':args.case,'variant':key,**report['variants'][key]}),flush=True)
    base=outputs['fixed_0.1'];checks={}
    for k,output in outputs.items():
        checks[k]={'same_time_grid':bool(np.array_equal(output[0],base[0])),'same_complete_td_mode':bool(np.array_equal(output[1],base[1])),'fd_relative_l2_to_fixed_0p1':float(np.linalg.norm(output[2]-base[2])/np.linalg.norm(base[2]))}
    report['invariance']=checks;after=source_hashes();report['source_hashes_after']=after
    before=report['source_hashes_before'];report['source_changes']=[p for p in before.keys()|after.keys() if before.get(p)!=after.get(p)]
    report['script_sha256']=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
    np.savez_compressed(folder/'tail-controls.npz',**arrays)
    (folder/'tail-controls.json').write_text(json.dumps(json_value(report),indent=2)+'\n')

if __name__=='__main__':main()
