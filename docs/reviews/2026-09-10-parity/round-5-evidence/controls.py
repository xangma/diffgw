"""Independent round-four controls. LAL is accessed through its public binary API only."""
import builtins
import copy
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import lal
from torchwave import get_fd_waveform
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS

ROOT = Path('/Users/xangma/repos/torchwave')
spec = importlib.util.spec_from_file_location('parity_harness_review', ROOT / 'tests/parity_harness.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
out = {}
case = copy.deepcopy(harness.DETERMINISTIC_HELD_OUT_CASES[0])
baseline = harness.run_oracle_comparison(case, strict=True)
out['baseline'] = baseline

original_waveform = harness.get_fd_waveform
for label, index in [('nonzero_dc', 0), ('nonzero_below_flow', 1), ('nonzero_above_fhigh', -1)]:
    def corrupt_bin(*args, **kwargs):
        hp, hc = original_waveform(*args, **kwargs)
        hp, hc = hp.clone(), hc.clone()
        hp[index], hc[index] = 1.0 + 1.0j, 1.0 - 1.0j
        return hp, hc
    with patch.object(harness, 'get_fd_waveform', corrupt_bin):
        result = harness.run_oracle_comparison(case, strict=True)
    out[label] = {k: result[k] for k in ['status', 'plus']}

original_metrics = harness.evaluate_waveform_metrics
for label, metric, value in [('bad_psd', 'psd_mismatch', .5),
                             ('bad_norm', 'relative_norm_error', .5),
                             ('nan_optimized_match', 'optimized_match', float('nan'))]:
    def corrupt_metric(*args, **kwargs):
        metrics = original_metrics(*args, **kwargs)
        metrics[metric] = value
        return metrics
    with patch.object(harness, 'evaluate_waveform_metrics', corrupt_metric):
        result = harness.run_oracle_comparison(case, strict=True)
    out[label] = {'status': result['status'], metric: str(value) if value != value else value}

original_projection = harness.project_strain
projection_calls = [0]
def corrupt_projection(*args, **kwargs):
    projection_calls[0] += 1
    strain = original_projection(*args, **kwargs)
    return strain * 2 if projection_calls[0] == 1 else strain
with patch.object(harness, 'project_strain', corrupt_projection):
    result = harness.run_oracle_comparison(case, strict=True)
out['doubled_test_projection'] = {'status': result['status'], 'projected_complex_relative_l2': result['detector_h1']['complex_relative_l2']}

with patch.object(harness, 'compute_antenna_factors', return_value=(.123, .456)):
    result = harness.run_oracle_comparison(case, strict=True)
out['wrong_shared_detector_geometry'] = {'status': result['status'], 'projected_complex_relative_l2': result['detector_h1']['complex_relative_l2']}

original_import = builtins.__import__
def without_pycbc(name, *args, **kwargs):
    if name == 'pycbc' or name.startswith('pycbc.'):
        raise ImportError('simulated absent PyCBC for strict dependency control')
    return original_import(name, *args, **kwargs)
with patch('builtins.__import__', without_pycbc):
    result = harness.run_oracle_comparison(case, strict=True)
out['strict_missing_pycbc'] = {'status': result['status'], 'optimized_match': result['plus']['optimized_match']}
with patch.dict(os.environ, {'STRICT_PARITY': '1'}), patch('builtins.__import__', without_pycbc):
    try:
        harness.test_parity_harness_held_out()
        out['pytest_strict_missing_pycbc'] = 'PASSED'
    except Exception as exc:
        out['pytest_strict_missing_pycbc'] = repr(exc)
with patch.object(harness, 'DETERMINISTIC_HELD_OUT_CASES', []):
    out['empty_suite'] = harness.run_full_parity_suite(strict=True)

out['detector_argument'] = {}
for name in ['H1', 'L1', 'V1', 'not-a-detector']:
    try:
        out['detector_argument'][name] = harness.compute_antenna_factors(1.2, .4, .3, detector=name)
    except Exception as exc:
        out['detector_argument'][name] = repr(exc)

f = torch.arange(20., 500., .25, dtype=torch.float64)
cfg = dict(mass1=30., mass2=20., f_ref=30., sample_frequencies=f)
out['wrapper_spin_aliases'] = {}
for name in ['TaylorF2', 'IMRPhenomD', 'TaylorF2_Eccentric']:
    result = {}
    try:
        batch_cfg = dict(cfg, mass1=torch.tensor([30.,30.]), mass2=torch.tensor([20.,20.]))
        h = get_fd_waveform(name, **batch_cfg, chi1=torch.tensor([.1,.2]), spin1z=torch.tensor([.1,.2]))
        result['matching_batch_alias_shape'] = list(h[0].shape)
    except Exception as exc:
        result['matching_batch_alias_error'] = str(exc)
    for label, extra in [('explicit_zero_conflict', dict(chi1=0.,spin1z=.3)), ('unknown_keyword', dict(misspelled_spin=.3))]:
        try:
            get_fd_waveform(name, **cfg, **extra)
            result[label] = 'accepted'
        except Exception as exc:
            result[label] = repr(exc)
    out['wrapper_spin_aliases'][name] = result

out['direct_model_spin_aliases'] = {}
for model in [TaylorF2(), TaylorF2_Eccentric(), IMRPhenomD(), IMRPhenomXAS()]:
    name = type(model).__name__
    result = {}
    for label, extra in [('spin1z_only', dict(spin1z=.3)), ('explicit_zero_conflict', dict(chi1=0.,spin1z=.3))]:
        try:
            h = model(f=f, mass1=30., mass2=20., **extra)
            result[label] = {'status':'accepted', 'finite':bool(torch.isfinite(h[0]).all())}
        except Exception as exc:
            result[label] = repr(exc)
    out['direct_model_spin_aliases'][name] = result

out['xphm_omitted_spin'] = {}
for label, extra in [('both_omitted', {}), ('chi1_only', dict(chi1=.2)), ('both_explicit',dict(chi1=0.,chi2=0.))]:
    try:
        h = get_fd_waveform('IMRPhenomXPHM', **cfg, **extra)
        out['xphm_omitted_spin'][label] = {'shape': list(h[0].shape), 'finite':bool(torch.isfinite(h[0]).all())}
    except Exception as exc:
        out['xphm_omitted_spin'][label] = repr(exc)

f = torch.arange(0.,500.25,.25,dtype=torch.float64)
ecc_cfg = dict(mass1=1.4,mass2=1.4,e0=.05,f_lower=30.,sample_frequencies=f)
canonical = get_fd_waveform('TaylorF2_Eccentric', **ecc_cfg)[0]
alias = get_fd_waveform('EccentricTaylorF2', **ecc_cfg)[0]
out['ecc_alias_equal'] = bool(torch.equal(alias, canonical))

common = dict(f=torch.arange(20.,512.,.25,dtype=torch.float64), mass1=35., mass2=30., chi1=.2,chi2=-.3,f_ref=20.,return_phase=True)
d = IMRPhenomD()(**common)
xas = IMRPhenomXAS()(**common)
out['xas_phase_exactly_d'] = bool(torch.equal(xas[2], d[2]))
out['xas_amplitude_exactly_d'] = bool(torch.equal(xas[0].abs(), d[0].abs()))
hm_cfg = dict(mass1=40.,mass2=20.,chi1=.3,chi2=-.1,inclination=1.0,sample_frequencies=common['f'])
hm = get_fd_waveform('IMRPhenomHM', **hm_cfg)
xhm = get_fd_waveform('IMRPhenomXHM', **hm_cfg)
out['hm_exactly_xhm'] = all(torch.equal(a,b) for a,b in zip(hm,xhm))
out['xas_batch_zero_passes_test_tolerance'] = bool(torch.allclose(xas[0], torch.zeros_like(xas[0]), rtol=1e-12, atol=1e-12))

Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
print(json.dumps(out, indent=2, allow_nan=False))
