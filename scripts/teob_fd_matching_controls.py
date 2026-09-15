#!/usr/bin/env python3
"""Factor NQC and ringdown matching policies on a frozen native trajectory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from teob_continuous_fd_attribution import OUT, case_for, evaluate_cached, json_value, source_hashes
from teob_cache_metadata import cache_fixed_step
from torchwave.approximants.eob.teob_merger_nqc_clean import assemble_teob_merger_nqc, published_2017_nqc_targets, PUBLISHED_2017_NQC_PROVENANCE
from torchwave.approximants.eob.teob_ringdown_assembly_clean import assemble_teob_ringdown_22
from torchwave.approximants.eob.teob_uib2016_remnant import uib2016_nonspinning_remnant
from torchwave.approximants.eob.teob_conditioned_waveform_clean import project_teob22_generic
from torchwave.approximants.eob.td_fd_conditioning_clean import condition_td_to_fd, plan_td_support

VARIANTS = {'grid':('grid','grid'), 'continuous':('continuous','continuous'),
            'continuous_nqc':('continuous','grid'), 'continuous_ringdown':('grid','continuous')}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',default='catalog')
    ap.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    ap.add_argument('--input-root',type=Path,default=OUT)
    ap.add_argument('--output-dir',type=Path,required=True,
                    help='Separate output root; existing frozen input reports are preserved.')
    ap.add_argument('--fixed-step',type=float,
                    help='Independently known integration step for legacy caches without scalar metadata.')
    args=ap.parse_args()
    path=args.input_root/args.case/'continuous.npz'
    out=args.output_dir/args.case
    if out.resolve()==path.parent.resolve():
        ap.error('--output-dir must differ from the frozen input root')
    out.mkdir(parents=True,exist_ok=True)
    data=np.load(path); case=case_for(args.case); p,g=case['params'],case['grid']
    tensor=lambda name:torch.as_tensor(data[name],dtype=torch.float64)
    m1,m2=p['mass1'],p['mass2']; nu=torch.tensor(m1*m2/(m1+m2)**2,dtype=torch.float64)
    final_mass,final_spin=uib2016_nonspinning_remnant(m1,m2)
    final_mass,final_spin=final_mass.double(),final_spin.double()
    support=plan_td_support(m1,m2,g['flow'],dtype=torch.float64)
    spacing,spacing_source=cache_fixed_step(data,args.fixed_step)
    before=source_hashes()
    report={'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'case':case,
            'source_hashes_before':before,'fixed_step_M':spacing,'fixed_step_source':spacing_source,
            'replay_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'metadata_helper_sha256':hashlib.sha256(Path(__file__).with_name('teob_cache_metadata.py').read_bytes()).hexdigest(),
            'variants':{},'notes':'Frozen accepted trajectory and original mode; no fitted alignment or coefficients. Mixed variants are attribution only.'}
    for variant in args.variants:
        start=time.time(); nmethod,rmethod=VARIANTS[variant]
        with torch.no_grad():
            nqc=assemble_teob_merger_nqc(tensor('nqc_original_time'),tensor('nqc_original_state'),
                amplitude=tensor('nqc_original_amplitude'),phase=tensor('nqc_original_phase'),
                orbital_frequency=tensor('nqc_original_orbital_frequency'),radial_acceleration=tensor('nqc_original_radial_acceleration'),
                nu=nu,fixed_step=spacing,targets=published_2017_nqc_targets(nu,torch.zeros_like(nu)),
                target_provenance=PUBLISHED_2017_NQC_PROVENANCE,matching_method=nmethod)
            # For the mixed ringdown control use the same independently measured
            # continuous event, even when NQC was solved on the old grid.
            peak=tensor('event_peak_time') if rmethod=='continuous' else None
            rd=assemble_teob_ringdown_22(nqc.original.time,nqc.original.corrected_amplitude,nqc.original.corrected_phase,
                nqc.auxiliary.time,nqc.auxiliary.corrected_amplitude,nqc.auxiliary.corrected_phase,nqc.auxiliary.orbital_frequency,
                nu,final_mass,final_spin,output_step=tensor('sampling_dt_seconds')/tensor('total_mass_seconds'),matching_method=rmethod,peak_time=peak)
            hp,hc=project_teob22_generic(rd.mode,torch.conj(rd.mode),mass1=m1,mass2=m2,distance=p['distance'],inclination=p['inclination'],phic=p['phic'])
            fd=condition_td_to_fd(hp,hc,dt=tensor('sampling_dt_seconds'),delta_f=g['df'],f_max=g['fhigh'],support=support,tc=0.,return_metadata=True)
        output=out/f'matching-{variant}.npz'
        np.savez_compressed(output,frequencies=fd.frequencies.numpy(),h_plus_fd=fd.h_plus.numpy(),h_cross_fd=fd.h_cross.numpy(),td_time=rd.time.numpy(),h_plus_td=hp.numpy(),h_cross_td=hc.numpy())
        item={'nqc_method':nmethod,'ringdown_method':rmethod,'nqc_coefficients':json_value(nqc.coefficients),
              'peak_M':float(nqc.t_peak),'rd_peak_M':float(rd.t_peak),'match_M':float(rd.t_match),
              'td_samples':hp.numel(),'conditioned_samples':fd.conditioning.h_plus.numel(),
              'cache_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'acceptance':evaluate_cached(fd.frequencies,fd.h_plus,fd.h_cross,case),'seconds':time.time()-start}
        if variant=='continuous':
            item['replay_vs_adapter']={pol:float(torch.linalg.vector_norm(getattr(fd,pol)-torch.as_tensor(data[pol+'_fd']))/torch.linalg.vector_norm(torch.as_tensor(data[pol+'_fd']))) for pol in ('h_plus','h_cross')}
        report['variants'][variant]=item
        print(json.dumps({'variant':variant,'plus':item['acceptance'].get('plus',{}).get('complex_relative_l2'),'cross':item['acceptance'].get('cross',{}).get('complex_relative_l2'),'seconds':item['seconds']}),flush=True)
        (out/'matching-controls.json').write_text(json.dumps(json_value(report),indent=2)+'\n')
    after=source_hashes();report['source_hashes_after']=after
    report['source_changes']=[p for p in before.keys()|after.keys() if before.get(p)!=after.get(p)]
    (out/'matching-controls.json').write_text(json.dumps(json_value(report),indent=2)+'\n')

if __name__=='__main__':main()
