#!/usr/bin/env python3
"""Evaluate the clean continuous TEOB adapter against the unchanged parity harness.

The diagnostic wrapper selects exact frequency bins and applies the public
waveform API's declared [flow,fhigh) band. It performs no fitted corrections,
alignment, interpolation, or threshold changes. All intermediate native stages
are cached for subsequent attribution without rerunning the evolution.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time

import lal
import lalsimulation as lalsim
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'scripts')]
import parity_harness as parity
from diagnose_teob_conditioned_waveform_clean import _cache_arrays
from torchwave.approximants.eob.teob_conditioned_waveform_clean import conditioned_teob_fd

OUT = ROOT / 'docs/reviews/2026-09-15-teob-fd-attribution'
CASES = {
    'catalog': (1.4, 1.4, 30., 500., .5),
    'bbh_reference': (30., 20., 20., 1024., .5),
    'equal_bbh': (20., 20., 25., 512., .5),
    'unequal_bbh': (40., 10., 20., 1024., .5),
    'unequal_bns': (1.6, 1.2, 30., 500., .5),
}


def json_value(x):
    if isinstance(x, torch.Tensor):
        y = x.detach().cpu()
        return y.item() if y.numel() == 1 else y.tolist()
    if dataclasses.is_dataclass(x):
        return {f.name: json_value(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict): return {k: json_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_value(v) for v in x]
    return x


def source_hashes():
    paths = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__), ROOT/'tests/parity_harness.py', ROOT/'scripts/diagnose_teob_conditioned_waveform_clean.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def case_for(name):
    case = copy.deepcopy(next(c for c in parity.get_canonical_catalog_cases() if c['id'] == 'TEOBResumS_diagnostic'))
    m1, m2, flow, fhigh, df = CASES[name]
    case['params'].update(mass1=m1, mass2=m2)
    case['grid'].update(flow=flow, fhigh=fhigh, df=df)
    if name != 'catalog': case['id'] = 'TEOB_continuous_' + name
    return case


def public_arrays(case):
    p, g = case['params'], case['grid']
    extra = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(extra, p['lambda1'])
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(extra, p['lambda2'])
    hp, hc = lalsim.SimInspiralFD(
        p['mass1']*lal.MSUN_SI, p['mass2']*lal.MSUN_SI,
        0., 0., 0., 0., 0., 0., p['distance']*1e6*lal.PC_SI,
        p['inclination'], p['phic'], 0., 0., 0.,
        g['df'], g['flow'], g['fhigh'], g['flow'], extra,
        lalsim.GetApproximantFromString('TEOBResumS'),
    )
    assert hp.f0 == hc.f0 == 0 and hp.deltaF == hc.deltaF == g['df']
    f = np.arange(hp.data.length, dtype=np.float64)*g['df']
    return f, np.array(hp.data.data), np.array(hc.data.data), {'hp_epoch':float(hp.epoch),'hc_epoch':float(hc.epoch)}


def band_wrapper(frequencies, hp, hc, case):
    """The production API band contract, applied without interpolating."""
    def generate(name, *, sample_frequencies, **kwargs):
        target = sample_frequencies.to(frequencies)
        idx = torch.searchsorted(frequencies, target)
        if bool((idx >= frequencies.numel()).any()) or not torch.equal(frequencies[idx], target):
            raise ValueError('candidate does not cover exact oracle frequency bins')
        mask = (target >= case['grid']['flow']) & (target < case['grid']['fhigh'])
        return torch.where(mask, hp[idx], 0.), torch.where(mask, hc[idx], 0.)
    return generate


def evaluate_cached(f, hp, hc, case):
    original = parity.get_fd_waveform
    parity.get_fd_waveform = band_wrapper(f, hp, hc, case)
    try:
        return parity.run_oracle_comparison(case, strict=True)
    finally:
        parity.get_fd_waveform = original


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=CASES, default='catalog')
    ap.add_argument('--matching-method', choices=('grid','continuous'), default='continuous')
    ap.add_argument('--output-dir', type=Path, default=OUT)
    args = ap.parse_args()
    out = args.output_dir/args.case
    out.mkdir(parents=True, exist_ok=True)
    report_path = out/f'{args.matching_method}.json'
    cache_path = out/f'{args.matching_method}.npz'
    case = case_for(args.case)
    started = time.time()
    report = {'status':'running','host':socket.gethostname(),'cwd':str(Path.cwd()),'pid':os.getpid(),
              'command':shlex.join([sys.executable,*sys.argv]),'started_unix':started,
              'case':case,'matching_method':args.matching_method,
              'source_hashes_before':source_hashes(),
              'band_contract':'Exact shared bins; native and oracle comparison [flow,fhigh), native out-of-band/DC zeroed by diagnostic API wrapper.',
              'policies':{'radial_acceleration_policy':'frozen-mobility','horizon_policy':'leading-order','pa_initializer_policy':'analytic'},
              'versions':{'torch':torch.__version__,'lal':lal.__version__,'lalsimulation':lalsim.__version__}}
    def save(): report_path.write_text(json.dumps(json_value(report),indent=2)+'\n')
    save()
    print(json.dumps({k:report[k] for k in ('status','host','cwd','pid','command')}),flush=True)
    try:
        p,g = case['params'],case['grid']
        with torch.no_grad():
            result = conditioned_teob_fd(p['mass1'],p['mass2'],f_low=g['flow'],f_max=g['fhigh'],delta_f=g['df'],distance=p['distance'],inclination=p['inclination'],phic=p['phic'],tc=p['tc'],matching_method=args.matching_method,**report['policies'],pa_kwargs={'r_min':14.,'dr':.1,'pa_order':8},dtype=torch.float64)
        report['native_seconds'] = time.time()-started
        report['support'] = json_value(result.support)
        report['sampling'] = json_value(result.sampling)
        report['native'] = {'td_samples':result.td_time.numel(),'conditioned_samples':result.conditioning.h_plus.numel(),'physical_epoch_seconds':float(result.physical_epoch_seconds),'accepted_peak_M':float(result.continuation.peak_event_time),'continuous_peak_M':float(result.nqc.t_peak),'matching_time_M':float(result.nqc.t_nqc),'pa_end_M':float(result.pa.time[-1]),'tail_end_M':float(result.td_time[-1]),'nqc_targets':json_value(result.nqc.targets),'nqc_coefficients':json_value(result.nqc.coefficients),'pa_statistics':json_value(result.pa.statistics) if hasattr(result.pa,'statistics') else None}
        arrays = _cache_arrays(result)
        arrays.update(conditioned_h_plus_td=result.conditioning.h_plus.numpy(),conditioned_h_cross_td=result.conditioning.h_cross.numpy())
        np.savez_compressed(cache_path,**arrays)
        report['cache']={'path':str(cache_path),'sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest()}
        report['status']='native_complete'; save()
        print(json.dumps({'status':report['status'],'native_seconds':report['native_seconds'],'samples':result.td_time.numel()}),flush=True)
        f,hp,hc,meta=public_arrays(case)
        oracle_path=out/'public-fd.npz'
        np.savez_compressed(oracle_path,frequencies=f,h_plus_fd=hp,h_cross_fd=hc)
        report['oracle']={**meta,'path':str(oracle_path),'sha256':hashlib.sha256(oracle_path.read_bytes()).hexdigest()}
        report['acceptance']=evaluate_cached(result.frequencies,result.h_plus,result.h_cross,case)
        report['status']='finished'
    except Exception as exc:
        report.update(status='error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds']=time.time()-started
        after=source_hashes()
        report['source_hashes_after']=after
        before=report['source_hashes_before']
        report['source_changes']=[p for p in before.keys()|after.keys() if before.get(p)!=after.get(p)]
        save()
    print(json.dumps({'status':report['status'],'acceptance_status':report['acceptance']['status'],'elapsed_seconds':report['elapsed_seconds'],'source_changes':report['source_changes']}),flush=True)
    return 0 if not report['source_changes'] else 1


if __name__=='__main__': raise SystemExit(main())
