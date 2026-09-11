"""Independent controls for the third parity review; uses the public binary oracle only."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torchwave import get_fd_waveform

ROOT = Path('/Users/xangma/repos/torchwave')
spec = importlib.util.spec_from_file_location('parity_harness_review', ROOT / 'tests/parity_harness.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
out = {}

case = copy.deepcopy(harness.DETERMINISTIC_HELD_OUT_CASES[0])
baseline = harness.run_oracle_comparison(case, strict=True)
out['baseline_support'] = {k: baseline[k] for k in ['status', 'plus', 'cross', 'detector_h1']}

original_waveform = harness.get_fd_waveform
def corrupt_dc(*args, **kwargs):
    hp, hc = original_waveform(*args, **kwargs)
    hp, hc = hp.clone(), hc.clone()
    hp[0], hc[0] = 1.0 + 1.0j, 1.0 - 1.0j
    return hp, hc
with patch.object(harness, 'get_fd_waveform', corrupt_dc):
    result = harness.run_oracle_comparison(case, strict=True)
out['nonzero_dc_negative_control'] = {'status': result['status'], 'plus_dc_zero': result['plus']['dc_zero'], 'cross_dc_zero': result['cross']['dc_zero']}

original_metrics = harness.evaluate_waveform_metrics
def corrupt_psd(*args, **kwargs):
    metrics = original_metrics(*args, **kwargs)
    metrics['psd_mismatch'] = 0.5
    return metrics
with patch.object(harness, 'evaluate_waveform_metrics', corrupt_psd):
    result = harness.run_oracle_comparison(case, strict=True)
out['psd_gate_negative_control'] = {'status': result['status'], 'psd_mismatch': result['plus']['psd_mismatch']}

original_projection = harness.project_strain
projection_calls = [0]
def corrupt_projection(*args, **kwargs):
    projection_calls[0] += 1
    strain = original_projection(*args, **kwargs)
    return strain * 2 if projection_calls[0] == 1 else strain
with patch.object(harness, 'project_strain', corrupt_projection):
    result = harness.run_oracle_comparison(case, strict=True)
out['projection_gate_negative_control'] = {'status': result['status'], 'projected_complex_relative_l2': result['detector_h1']['complex_relative_l2']}

original_import = builtins.__import__
def without_pycbc(name, *args, **kwargs):
    if name == 'pycbc' or name.startswith('pycbc.'):
        raise ImportError('simulated absent PyCBC for strict dependency control')
    return original_import(name, *args, **kwargs)
with patch('builtins.__import__', without_pycbc):
    result = harness.run_oracle_comparison(case, strict=True)
out['strict_missing_pycbc'] = {'status': result['status'], 'optimized_match': result['plus']['optimized_match']}

out['detector_argument'] = {name: harness.compute_antenna_factors(1.2, .4, .3, detector=name) for name in ['H1', 'L1', 'V1', 'not-a-detector']}

f = torch.arange(20., 500., .25, dtype=torch.float64)
cfg = dict(mass1=30., mass2=20., f_ref=30., sample_frequencies=f)
out['spin_alias_edges'] = {}
for name in ['TaylorF2', 'IMRPhenomD', 'TaylorF2_Eccentric']:
    result = {}
    try:
        batch_cfg = dict(cfg, mass1=torch.tensor([30.,30.]), mass2=torch.tensor([20.,20.]))
        h = get_fd_waveform(name, **batch_cfg, chi1=torch.tensor([.1,.2]), spin1z=torch.tensor([.1,.2]))
        result['matching_batch_alias'] = {'shape': list(h[0].shape)}
    except Exception as exc:
        result['matching_batch_alias'] = {'error': str(exc)}
    try:
        get_fd_waveform(name, **cfg, chi1=0., spin1z=.3)
        result['explicit_zero_conflict_rejected'] = False
    except ValueError:
        result['explicit_zero_conflict_rejected'] = True
    try:
        a = get_fd_waveform(name, **cfg)[0]
        b = get_fd_waveform(name, **cfg, misspelled_spin=.3)[0]
        result['unknown_keyword_ignored'] = bool(torch.equal(a,b))
    except (ValueError, TypeError):
        result['unknown_keyword_ignored'] = False
    out['spin_alias_edges'][name] = result

h = get_fd_waveform('TaylorF2', **cfg, spin1z=.3)[0]
out['existing_alias_test_tolerance_control'] = {'max_waveform_magnitude': float(h.abs().max()), 'zero_waveform_passes_existing_allclose': bool(torch.allclose(h, torch.zeros_like(h), atol=1e-15))}

f = torch.arange(0.,500.25,.25,dtype=torch.float64)
ecc_cfg = dict(mass1=1.4,mass2=1.4,e0=.05,f_lower=30.,sample_frequencies=f)
canonical = get_fd_waveform('TaylorF2_Eccentric', **ecc_cfg)[0]
alias = get_fd_waveform('EccentricTaylorF2', **ecc_cfg)[0]
explicit20 = get_fd_waveform('TaylorF2_Eccentric', **ecc_cfg, f0_ecc=20.)[0]
sel = f >= 30.
out['ecc_model_alias_reference'] = {'relative_l2_vs_canonical': float(torch.linalg.norm(alias[sel]-canonical[sel])/torch.linalg.norm(canonical[sel])), 'alias_equals_explicit_f0_20': bool(torch.equal(alias,explicit20))}

path = Path(__file__).with_suffix('.json')
path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
