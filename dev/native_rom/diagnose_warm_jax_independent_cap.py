"""Dev-only independent Torch/JAX cap evolution with matched historical ODE attachment."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom.diagnose_source_aligned_native_cap_full_crop_v5 import final_h22_crop
from dev.native_rom.diagnose_source_aligned_ode_chain_amplitude_v2 import _amplitude_hook, _continuous_amplitude
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _phase_matrix
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_rhs import _rhs_derivatives
from dev.native_rom.eob.jax import numerical_attachment_clean as ja
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from torchwave.approximants.eob import numerical_attachment as ta
from torchwave.approximants.eob.adaptive_driver_clean import AlignedSpinV4Event, integrate_adaptive

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'dev/native_rom/evidence/2026-09-25'
PLAN = EVIDENCE / 'warm_jax_independent_cap_plan.json'
OUTPUT = EVIDENCE / 'warm_jax_independent_cap.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def compare(left, right):
    a, b = arr(left), arr(right)
    n = min(len(a), len(b))
    if n < 1:
        raise ValueError('empty common comparison')
    return {'left_samples': len(a), 'right_samples': len(b), 'common_samples': n,
            'relative_l2': float(np.linalg.norm(a[:n]-b[:n])/np.linalg.norm(b[:n])),
            'max_abs': float(np.max(abs(a[:n]-b[:n])))}


def main():
    plan = json.loads(PLAN.read_text())
    if (plan['schema'] != 'diffgw.seobnrv4.warm-jax-independent-cap-plan.v1'
            or plan['release_qualified'] is not False
            or sha(__file__) != plan['script_sha256']):
        raise ValueError('frozen plan/runner changed')
    for item in plan['inputs'].values():
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError(f"frozen input changed: {item['path']}")
    cap = json.loads((ROOT/plan['inputs']['cap_v5']['path']).read_text())
    old = cap['levels']['refined']['ringdown_boundary_controls']
    if (cap['status'] != 'diagnostic_complete' or cap['cap_state'] != plan['cap_state']
            or cap['row_tuple'] != plan['row_tuple']
            or old['continuous_frequency'] != plan['historical']['frequency']
            or old['phase_coefficients'] != plan['historical']['phase_coefficients']
            or old['match_time_local_M'] != plan['historical']['match_time_local_M']):
        raise ValueError('historical cap evidence changed')
    torch.set_num_threads(1)
    m1, m2, chi1, chi2, requested = plan['row_tuple']
    nu, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
    def tensor(x):
        return torch.tensor(x, dtype=torch.float64)
    th = driver.TorchSourceHam(tensor(nu), tensor(chi1), tensor(chi2), delta=tensor(delta))
    tf = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    jh = driver.JaxSourceHam(nu, chi1, chi2, delta=delta)
    jf = driver.JaxFlux(**driver.SOURCE_FLUX_OPTIONS)
    tfield = driver.TorchIntegrator(th, tf).vector_field
    jfield = jax.jit(driver.JaxIntegrator(jh, jf).vector_field)
    state = np.asarray(plan['cap_state'], dtype=np.float64)
    ts, js = torch.as_tensor(state), jnp.asarray(state)
    s = plan['settings']
    dt = s['coarse_output_step_M']
    result = {'schema': 'diffgw.seobnrv4.warm-jax-independent-cap.v1',
              'status': 'running', 'release_qualified': False, 'host': platform.node(),
              'cwd': os.getcwd(), 'pid': os.getpid(), 'script_sha256': sha(__file__),
              'plan_sha256': sha(PLAN), 'row_tuple': plan['row_tuple'],
              'settings': s, 'stages': {}, 'checks': {}}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    started = time.monotonic()
    result['stages']['initial'] = {'state_max_abs': float(np.max(abs(arr(ts)-arr(js)))),
        'field_max_abs': float(np.max(abs(arr(tfield(ts))-arr(jfield(js)))))}
    tc = integrate_adaptive(tfield, ts, dt, t_end=s['max_duration_M'],
        event=AlignedSpinV4Event('coarse'), initial_step=dt,
        max_step=s['max_step_M'], controller='gsl_state_scaled',
        atol=s['atol'], rtol=s['rtol'])
    print('Torch coarse done', flush=True)
    jr = integrate_two_stage_trajectory(jfield, js, output_step=dt,
        max_duration=s['max_duration_M'], coarse_initial_step=dt,
        coarse_max_step=s['max_step_M'], fine_max_step=s['max_step_M'],
        fine_refinement_factor=s['fine_refinement_factor'],
        atol=s['atol'], rtol=s['rtol'])
    print('JAX coarse/fine done', flush=True)
    restart = len(tc.output_time)-math.ceil(150/dt)
    fine_dt = dt/s['fine_refinement_factor']
    tfr = integrate_adaptive(tfield, tc.output_state[restart], fine_dt,
        t_end=s['max_duration_M'], event=AlignedSpinV4Event('fine'),
        initial_step=fine_dt, max_step=s['max_step_M'],
        controller='gsl_state_scaled', atol=s['atol'], rtol=s['rtol'])
    print('Torch fine done', flush=True)
    ct, cs, ft, fs = tc.output_time, tc.output_state, tfr.output_time, tfr.output_state
    jct, jcs, jft, jfs = jr.coarse_time, jr.coarse_state, jr.fine_time, jr.fine_state
    result['stages']['trajectory'] = {
        'torch_restart_index': restart, 'jax_restart_index': jr.restart_index,
        'torch_coarse_samples': len(ct), 'jax_coarse_samples': len(jct),
        'torch_fine_samples': len(ft), 'jax_fine_samples': len(jft),
        'coarse_time': compare(jct, ct), 'coarse_state': compare(jcs, cs),
        'fine_time': compare(jft, ft), 'fine_state': compare(jfs, fs),
        'restart_state_max_abs': float(np.max(abs(arr(jfs[0])-arr(fs[0])))),
        'torch_coarse_event_time_M': float(tc.time[-1]),
        'jax_coarse_event_time_M': float(arr(jr.coarse.time)[-1]),
        'torch_fine_event_time_M': float(tfr.time[-1]),
        'jax_fine_event_time_M': float(arr(jr.fine.time)[-1]),
    }
    topt = driver.MODE_OPTIONS | {'retain_pan_cubic_spin': True}
    two = tfield(fs)[:, 1]
    jfo = jax.vmap(jfield)(jfs)[:, 1]
    tco = tfield(cs)[:, 1]
    jco = jax.vmap(jfield)(jcs)[:, 1]
    traw, trf = driver.torch_mode(fs, th, tf, **topt)
    jmode_fn = jax.jit(lambda x: driver.jax_mode(x, jh, jf, **topt))
    jraw, jrf = jmode_fn(jfs)
    jsame, _ = jmode_fn(jnp.asarray(arr(fs)))
    tcraw, _ = driver.torch_mode(cs, th, tf, **topt)
    jcraw, _ = jmode_fn(jcs)
    result['stages']['raw'] = {
        'fine_same_states': compare(jsame, traw),
        'fine_own_states': compare(jraw, traw),
        'coarse_own_states': compare(jcraw, tcraw),
        'fine_frequency_own': compare(jrf, trf),
    }
    match = tensor(plan['historical']['match_time_local_M'])
    phase_capture, rhs_capture = [], []
    phase_matrix = _phase_matrix(ft, fs, two, match,
        half_width_M=0.0, degree=0, field=tfield, capture=phase_capture)
    center = phase_capture[0]['interpolated_state']
    amplitude_rows, amplitude_raw = _continuous_amplitude(center, th, tf, tfield)
    rhs = _rhs_derivatives(ft, traw, match, center_state=center,
        hamiltonian=th, flux=tf, field=tfield, capture=rhs_capture)
    rhs[0, :] = amplitude_rows[1:, 0]
    amp_deriv = amplitude_rows[1:, 1:4]
    tnu, tchi = driver.torch_parameters(*map(tensor, (m1,m2,chi1,chi2)))
    jnu, jchi = driver.jax_parameters(m1,m2,chi1,chi2)
    tmass = driver.torch_final_mass(*map(tensor,(m1,m2,chi1,chi2)))
    jmass = driver.jax_final_mass(m1,m2,chi1,chi2)
    tspin = driver.torch_final_spin(*map(tensor,(m1,m2,chi1,chi2)), fit_order=(3,4))
    jspin = driver.jax_final_spin(m1,m2,chi1,chi2,fit_order=(3,4))
    tw,tdamp = driver.torch_qnm(tspin)
    jw,jdamp = driver.jax_qnm(jspin)
    tsigma = torch.complex(tw,-tdamp)/tmass
    jsigma = (jw-1j*jdamp)/jmass
    restart_time = float(ct[restart])
    output_end = cap['levels']['refined']['full_series']['end_M']-restart_time
    output = torch.arange(round(output_end/fine_dt)+1,dtype=torch.float64)*fine_dt
    original_tfull, original_teval = ta._full_series_nqc, ta._eval_batch_scalar
    original_jcal = ja.calibrate_nqc_22_full_series
    original_tring, original_jring = ta.merger_ringdown_22, ja.merger_ringdown_22
    hooks = {'torch_amplitude': [], 'jax_amplitude': [], 'ringdown': {}}
    def jcal(*args, **kwargs):
        nqc = original_jcal(*args, **kwargs)
        matrix = jnp.asarray(arr(amplitude_rows[:, 1:4]))
        targets = jnp.stack(args[7][:3])
        rhs_amp = targets-jnp.asarray(arr(amplitude_rows[:,0]))
        a, cond, residual = ja._solve_checked(matrix, rhs_amp,
            kwargs.get('max_condition',1e10), 'historical cap ODE amplitude')
        cal = nqc.calibration._replace(
            coefficients=nqc.calibration.coefficients._replace(amplitude=a),
            amplitude_condition=cond, amplitude_residual=residual)
        corrected = nqc.sampled_amplitude+jnp.sum(nqc.sampled_products*a,axis=-1)
        if not bool(jnp.all(corrected>0)):
            raise ValueError('nonpositive corrected cap amplitude')
        corrected_mode = corrected*jnp.exp(-1j*(nqc.sampled_waveform_phase-
            jnp.sum(nqc.sampled_phase_basis*cal.coefficients.phase,axis=-1)))
        hooks['jax_amplitude'].append({'coefficients':arr(a).tolist(),
            'matrix':arr(matrix).tolist(),'rhs':arr(rhs_amp).tolist()})
        return replace(nqc,calibration=cal,amplitude_matching_matrix=matrix,
            sampled_corrected_amplitude=corrected,
            sampled_corrected_phase=ja.unwrap_mode_phase(corrected_mode))
    def tring(elapsed,sigma,amplitude,amplitude_derivative,frequency,phase,shape):
        hooks['ringdown']['torch_original'] = float(frequency)
        return original_tring(elapsed,sigma,amplitude,amplitude_derivative,
            tensor(plan['historical']['frequency']),phase,shape)
    def jring(elapsed,sigma,amplitude,amplitude_derivative,frequency,phase,shape):
        hooks['ringdown']['jax_original'] = float(frequency)
        return original_jring(elapsed,sigma,amplitude,amplitude_derivative,
            jnp.asarray(plan['historical']['frequency']),phase,shape)
    with (patch.object(ta,'_full_series_nqc',_amplitude_hook(
            original_tfull,original_teval,amplitude_rows,hooks['torch_amplitude'])),
          patch.object(ta,'merger_ringdown_22',tring)):
        tref = driver.torch_attach(ft,fs,two,traw,trf,tnu,tchi,tsigma,output,
            match_time=match,diagnostic_amplitude_derivatives=amp_deriv,
            diagnostic_phase_matrix=phase_matrix,
            diagnostic_matching_rhs_derivatives=rhs,diagnostic_continuous_boundary=True)
    with (patch.object(ja,'calibrate_nqc_22_full_series',jcal),
          patch.object(ja,'merger_ringdown_22',jring)):
        jref = driver.jax_attach(jft,jfs,jfo,jraw,jrf,jnu,jchi,jsigma,
            jnp.asarray(arr(output)),match_time=jnp.asarray(arr(match)),
            diagnostic_amplitude_derivatives=jnp.asarray(arr(amp_deriv)),
            diagnostic_phase_matrix=jnp.asarray(arr(phase_matrix)),
            diagnostic_matching_rhs_derivatives=jnp.asarray(arr(rhs)),
            diagnostic_continuous_boundary=True)
    tcoef,jcoef=tref.calibration.coefficients,jref.calibration.coefficients
    boundary_index=int(tref.boundary_index)
    with torch.enable_grad():
        bs=fs[boundary_index].detach().clone().requires_grad_(True)
        vector=tfield(bs)
        _,raw_freq=driver.torch_mode(bs,th,tf,**topt)
        basis=driver.torch_basis(bs[0],bs[2],vector[1])[1]
        basis_first=torch.stack(tuple(
            (torch.autograd.grad(basis[i],bs,retain_graph=True)[0]*vector).sum()
            for i in range(2)))
        ode_freq=float((raw_freq-basis_first.dot(tcoef.phase.detach())).detach())
    tmode,jmode=arr(tref.mode),arr(jref.mode)
    before=arr(output)<float(match)
    after=~before
    result['stages']['attachment'] = {
        'match_time_M':float(match),'torch_boundary_index':boundary_index,
        'jax_boundary_index':int(jref.boundary_index),
        'boundary_time_abs_M':abs(float(tref.boundary_time)-float(jref.boundary_time)),
        'historical_phase_coeff_max_abs':float(np.max(abs(arr(tcoef.phase)-
            np.asarray(plan['historical']['phase_coefficients'])))),
        'historical_ode_frequency_abs':abs(ode_freq-plan['historical']['frequency']),
        'ode_frequency_computed':ode_freq,
        'amplitude_coeff_max_abs':float(np.max(abs(arr(tcoef.amplitude)-arr(jcoef.amplitude)))),
        'phase_coeff_max_abs':float(np.max(abs(arr(tcoef.phase)-arr(jcoef.phase)))),
        'corrected_inspiral':compare(jref.corrected_inspiral,tref.corrected_inspiral),
        'pre_boundary_mode':compare(jmode[before],tmode[before]),
        'post_boundary_mode':compare(jmode[after],tmode[after]),
        'fine_whole_mode':compare(jmode,tmode),
        'controls':{'amplitude_rows':arr(amplitude_rows).tolist(),
            'amplitude_raw':amplitude_raw,'phase_matrix':arr(phase_matrix).tolist(),
            'rhs':arr(rhs).tolist(),'phase_capture':phase_capture[0],
            'rhs_capture':rhs_capture[0], 'hooks':hooks},
    }
    tnqc=driver.torch_factor(cs[:,0],cs[:,2],tco,tcoef)
    jnqc=driver.jax_factor(jcs[:,0],jcs[:,2],jco,jcoef)
    tprefix=arr(tcraw[:restart]*tnqc[:restart])
    jprefix=arr(jcraw[:jr.restart_index]*jnqc[:jr.restart_index])
    tfull=np.concatenate((tprefix,tmode[::s['fine_refinement_factor']]))
    jfull=np.concatenate((jprefix,jmode[::s['fine_refinement_factor']]))
    ttime=np.concatenate((arr(ct[:restart]),restart_time+arr(output)[::s['fine_refinement_factor']]))
    jtime=np.concatenate((arr(jct[:jr.restart_index]),
        float(jct[jr.restart_index])+arr(output)[::s['fine_refinement_factor']]))
    tcrop=final_h22_crop(ttime,tfull,requested)
    jcrop=final_h22_crop(jtime,jfull,requested)
    crop_index=tcrop['index']
    result['stages']['whole']={'torch_samples':len(tfull),'jax_samples':len(jfull),
        'time':compare(jtime,ttime),'coarse_prefix':compare(jprefix,tprefix),
        'assembled':compare(jfull,tfull),
        'common_after_torch_crop':compare(jfull[crop_index:],tfull[crop_index:]),
        'torch_crop':tcrop,'jax_crop':jcrop,
        'crop_index_difference':jcrop['index']-tcrop['index']}
    g=plan['gates']; st=result['stages']; tr=st['trajectory']; raw=st['raw']
    at=st['attachment']; wh=st['whole']
    result['checks']={
        'initial_state':st['initial']['state_max_abs']<=g['initial_state_max_abs'],
        'initial_field':st['initial']['field_max_abs']<=g['initial_field_max_abs'],
        'restart_index':tr['torch_restart_index']==tr['jax_restart_index'],
        'coarse_time':tr['coarse_time']['max_abs']<=g['time_max_abs_M'],
        'fine_time':tr['fine_time']['max_abs']<=g['time_max_abs_M'],
        'historical_match':abs(float(match)-old['match_time_local_M'])<=g['time_max_abs_M'],
        'historical_phase':at['historical_phase_coeff_max_abs']<=g['historical_control_max_abs'],
        'historical_frequency':at['historical_ode_frequency_abs']<=g['historical_control_max_abs'],
        'same_state_raw':raw['fine_same_states']['relative_l2']<=g['same_state_raw_l2'],
        'own_state_raw':raw['fine_own_states']['relative_l2']<=g['own_state_raw_l2'],
        'nqc_coefficients':max(at['amplitude_coeff_max_abs'],at['phase_coeff_max_abs'])<=g['nqc_coefficient_max_abs'],
        'same_boundary':at['boundary_time_abs_M']<=g['time_max_abs_M'] and
            at['torch_boundary_index']==at['jax_boundary_index'],
        'corrected_inspiral':at['corrected_inspiral']['relative_l2']<=g['whole_mode_l2'],
        'pre_boundary':at['pre_boundary_mode']['relative_l2']<=g['whole_mode_l2'],
        'post_boundary':at['post_boundary_mode']['relative_l2']<=g['whole_mode_l2'],
        'coarse_prefix':wh['coarse_prefix']['relative_l2']<=g['whole_mode_l2'],
        'assembled_whole':wh['assembled']['relative_l2']<=g['whole_mode_l2'],
        'cropped_whole':wh['common_after_torch_crop']['relative_l2']<=g['whole_mode_l2'],
        'crop_index':wh['crop_index_difference']==0,
        'full_chart':wh['torch_samples']==wh['jax_samples']==cap['levels']['refined']['full_series']['samples']
            and wh['time']['max_abs']<=g['time_max_abs_M'],
        'strict_crops':tcrop['strict_first_interior_crossing'] and jcrop['strict_first_interior_crossing'],
        'hooks_reached':len(hooks['torch_amplitude'])==len(hooks['jax_amplitude'])==1
            and set(hooks['ringdown'])=={'torch_original','jax_original'},
    }
    result['diagnostic_checks_passed']=all(result['checks'].values())
    result['status']='diagnostic_complete' if result['diagnostic_checks_passed'] else 'failed_parity'
    result['elapsed_s']=time.monotonic()-started
    OUTPUT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        prior=json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
        prior.update({'status':'failed_execution','release_qualified':False,
                      'error':repr(error),'traceback':traceback.format_exc()})
        OUTPUT.write_text(json.dumps(prior,indent=2,sort_keys=True,allow_nan=False)+'\n')
        raise
