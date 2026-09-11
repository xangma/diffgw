"""Independent round-five controls; LAL public binary API only."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import lal
import torch
from torchwave import get_fd_waveform
from torchwave.detectors.network import Detector, greenwich_mean_sidereal_time

ROOT = Path('/Users/xangma/repos/torchwave')
spec = importlib.util.spec_from_file_location('round5_harness', ROOT / 'tests/parity_harness.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
out = {}
f = torch.arange(20., 512., .25, dtype=torch.float64)
cfg = dict(mass1=35., mass2=30., chi1=.2, chi2=-.3, f_ref=20., sample_frequencies=f)
out['execution_contexts'] = {}
for model in ['IMRPhenomD', 'IMRPhenomXAS', 'IMRPhenomHM', 'IMRPhenomXHM', 'IMRPhenomXPHM']:
    row = {}
    for label, context in [('normal', torch.enable_grad), ('no_grad', torch.no_grad), ('inference_mode', torch.inference_mode)]:
        try:
            with context():
                hp, hc = get_fd_waveform(model, **cfg)
            row[label] = {'finite': bool(torch.isfinite(hp).all() and torch.isfinite(hc).all()), 'shape': list(hp.shape)}
        except Exception as e:
            row[label] = repr(e)
    out['execution_contexts'][model] = row

out['xas_unknown_keyword'] = {}
for key in ['misspelled_spin', 'unrecognized_option']:
    try:
        a = get_fd_waveform('IMRPhenomXAS', **cfg)
        b = get_fd_waveform('IMRPhenomXAS', **cfg, **{key:.7})
        out['xas_unknown_keyword'][key] = {'accepted': True, 'identical': all(torch.equal(x,y) for x,y in zip(a,b))}
    except Exception as e:
        out['xas_unknown_keyword'][key] = repr(e)

case = copy.deepcopy(harness.DETERMINISTIC_HELD_OUT_CASES[0])
with patch.object(harness, 'DETERMINISTIC_HELD_OUT_CASES', []), patch.dict(os.environ, {'STRICT_PARITY':'1'}):
    try:
        harness.test_parity_harness_held_out()
        out['pytest_empty_suite'] = 'PASSED'
    except Exception as e:
        out['pytest_empty_suite'] = repr(e)

with patch.object(harness, 'DETERMINISTIC_HELD_OUT_CASES', [copy.deepcopy(case) for _ in range(6)]):
    result = harness.run_full_parity_suite(strict=True)
    out['six_duplicate_cases'] = {k:result[k] for k in ['suite_passed','num_cases','num_passed']}
    out['six_duplicate_cases']['case_ids'] = [r.get('case_id') for r in result['results']]

with patch.object(harness, 'aLIGOZeroDetHighPower', side_effect=lambda f:torch.full_like(f, float('nan'))):
    result = harness.run_oracle_comparison(case, strict=True)
    out['nan_psd_array'] = {'status':result['status'], 'reported_psd_mismatch':result['plus']['psd_mismatch']}

orig_metrics = harness.evaluate_waveform_metrics
def negative_infinite_metric(*args,**kwargs):
    metrics = orig_metrics(*args,**kwargs)
    metrics['psd_mismatch'] = float('-inf')
    return metrics
with patch.object(harness, 'evaluate_waveform_metrics', side_effect=negative_infinite_metric):
    result = harness.run_oracle_comparison(case, strict=True)
    out['negative_infinite_psd_metric'] = {'status':result['status']}

out['detector_public_oracle'] = []
for gps in [1126259462.,1126259462.+86400.,1187008882.,1470000000.]:
    gps_lal = lal.LIGOTimeGPS(gps)
    gmst_lal = lal.GreenwichMeanSiderealTime(gps_lal)
    gmst_tw = float(greenwich_mean_sidereal_time(torch.tensor(gps,dtype=torch.float64)))
    row = {'gps':gps,'wrapped_gmst_error_radians':math.atan2(math.sin(gmst_tw-gmst_lal),math.cos(gmst_tw-gmst_lal)), 'detectors':{}}
    for name,index in [('H1',lal.LALDetectorIndexLHODIFF),('L1',lal.LALDetectorIndexLLODIFF),('V1',lal.LALDetectorIndexVIRGODIFF)]:
        d=Detector(name)
        fp,fc=d.antenna_patterns(1.2,.4,.3,gps)
        dt=d.time_delay_from_earth_center(1.2,.4,gps)
        ref=lal.CachedDetectors[index]
        rfp,rfc=lal.ComputeDetAMResponse(ref.response,1.2,.4,.3,gmst_lal)
        rdt=lal.TimeDelayFromEarthCenter(ref.location,1.2,.4,gps_lal)
        row['detectors'][name]={'fp_error':float(fp)-rfp,'fc_error':float(fc)-rfc,'delay_error_seconds':float(dt)-rdt}
    out['detector_public_oracle'].append(row)

Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,indent=2,allow_nan=False))
