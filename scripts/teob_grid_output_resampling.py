#!/usr/bin/env python3
"""Isolate accepted-versus-joined output interpolation before grid attachment."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shlex
import socket
import sys
import time
import numpy as np
import torch
from teob_grid_derivative_controls import (
    OUT, INPUT, CASES, source_hashes, case_for, digest, json_value,
    assemble_teob_merger_nqc, published_2017_nqc_targets, PUBLISHED_2017_NQC_PROVENANCE,
    assemble_teob_ringdown_22, uib2016_nonspinning_remnant, natural_cubic_spline,
    project_teob22_generic, condition_td_to_fd, plan_td_support, evaluate_cached,
)
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=CASES,required=True)
    args=ap.parse_args()
    out=OUT/args.case
    case=case_for(args.case); p,g=case['params'],case['grid']
    cache=INPUT/args.case/'continuous.npz'
    z=np.load(cache,allow_pickle=False)
    t=lambda k:torch.as_tensor(z[k],dtype=torch.float64)
    m1,m2=p['mass1'],p['mass2'];nu=torch.tensor(m1*m2/(m1+m2)**2,dtype=torch.float64)
    report={'status':'running','case':case,'host':socket.gethostname(),'cwd':str(Path.cwd()),
            'pid':os.getpid(),'command':shlex.join([sys.executable,*sys.argv]),
            'input_sha256':digest(cache),'nominal_sha256':digest(out/'nominal.npz'),
            'script_sha256':digest(Path(__file__)),'protocol_sha256':digest(OUT/'resampling-protocol.md'),
            'source_hashes_before':source_hashes(),'variants':{}}
    path=out/'output-resampling.json'
    def save():path.write_text(json.dumps(json_value(report),indent=2)+'\n')
    save(); print(json.dumps({k:report[k] for k in ('host','cwd','pid','command')}),flush=True)
    begun=time.time()
    try:
        with torch.no_grad():
            nqc=assemble_teob_merger_nqc(t('nqc_original_time'),t('nqc_original_state'),
                amplitude=t('nqc_original_amplitude'),phase=t('nqc_original_phase'),
                orbital_frequency=t('nqc_original_orbital_frequency'),radial_acceleration=t('nqc_original_radial_acceleration'),
                nu=nu,fixed_step=.1,targets=published_2017_nqc_targets(nu,torch.zeros_like(nu)),
                target_provenance=PUBLISHED_2017_NQC_PROVENANCE,matching_method='grid')
            mf,spin=[v.double() for v in uib2016_nonspinning_remnant(m1,m2)]
            rd=assemble_teob_ringdown_22(nqc.original.time,nqc.original.corrected_amplitude,nqc.original.corrected_phase,
                nqc.auxiliary.time,nqc.auxiliary.corrected_amplitude,nqc.auxiliary.corrected_phase,nqc.auxiliary.orbital_frequency,
                nu,mf,spin,output_step=t('sampling_dt_seconds')/t('total_mass_seconds'),matching_method='grid')
            pre=rd.time<rd.events.attachment_time
            accepted_amplitude=natural_cubic_spline(nqc.original.time,nqc.original.corrected_amplitude)(rd.time[pre])
            accepted_phase=natural_cubic_spline(nqc.original.time,nqc.original.corrected_phase)(rd.time[pre])
            support=plan_td_support(m1,m2,g['flow'],dtype=torch.float64)
            nominal=np.load(out/'nominal.npz',allow_pickle=False)
            report.update(pre_samples=int(pre.sum()),post_samples=int((~pre).sum()),attachment_time_M=rd.events.attachment_time,
                          pre_max_amplitude_change=(accepted_amplitude-rd.amplitude[pre]).abs().max(),
                          pre_max_phase_change=(accepted_phase-rd.phase[pre]).abs().max())
            baseline_fd=None
            for variant in ('baseline','accepted_amplitude','accepted_phase','accepted_both'):
                mode=rd.mode.clone()
                if variant!='baseline':
                    a=accepted_amplitude if variant in ('accepted_amplitude','accepted_both') else rd.amplitude[pre]
                    ph=accepted_phase if variant in ('accepted_phase','accepted_both') else rd.phase[pre]
                    mode[pre]=a*torch.exp(torch.complex(torch.zeros_like(ph),-ph))
                assert torch.equal(mode[~pre],rd.mode[~pre])
                hp,hc=project_teob22_generic(mode,torch.conj(mode),mass1=m1,mass2=m2,distance=p['distance'],inclination=p['inclination'],phic=p['phic'])
                fd=condition_td_to_fd(hp,hc,dt=t('sampling_dt_seconds'),delta_f=g['df'],f_max=g['fhigh'],support=support,tc=p['tc'],return_metadata=True)
                detail={'post_attachment_mode_bitwise':True}
                if variant=='baseline':
                    detail['nominal_bitwise']=all(np.array_equal(v.numpy(),nominal[k]) for k,v in
                        [('td_time',rd.time),('h_plus_td',hp),('h_cross_td',hc),('h_plus_fd',fd.h_plus),('h_cross_fd',fd.h_cross)])
                    assert detail['nominal_bitwise']
                    baseline_fd=fd
                detail['fd_change_vs_nominal']={pol:torch.linalg.vector_norm(getattr(fd,pol)-getattr(baseline_fd,pol))/torch.linalg.vector_norm(getattr(baseline_fd,pol)) for pol in ('h_plus','h_cross')}
                detail['acceptance']=evaluate_cached(fd.frequencies,fd.h_plus,fd.h_cross,case)
                output=out/f'output-{variant}.npz'
                np.savez_compressed(output,frequencies=fd.frequencies.numpy(),h_plus_fd=fd.h_plus.numpy(),h_cross_fd=fd.h_cross.numpy(),td_time=rd.time.numpy(),h_plus_td=hp.numpy(),h_cross_td=hc.numpy())
                detail['output_sha256']=digest(output)
                report['variants'][variant]=detail;save()
                print(json.dumps({'case':args.case,'variant':variant,'plus':detail['acceptance']['plus']['complex_relative_l2'],'cross':detail['acceptance']['cross']['complex_relative_l2']}),flush=True)
        report['status']='finished'
    except Exception as exc:
        report.update(status='error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds']=time.time()-begun
        report['source_hashes_after']=source_hashes()
        report['source_changes']=[p for p,h in report['source_hashes_before'].items() if report['source_hashes_after'].get(p)!=h]
        save()
    assert not report['source_changes']


if __name__=='__main__':main()
