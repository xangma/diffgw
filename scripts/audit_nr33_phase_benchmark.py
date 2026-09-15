"""Independent NumPy/SciPy replay of the saved NR33 phase benchmark.

This intentionally does not import the benchmark metric implementation or call
LAL. It consumes only report.json and the saved NPZ arrays.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar


def metrics(f, candidate, reference, weights, phase_mask):
    f = np.asarray(f, float); c = np.asarray(candidate, complex); r = np.asarray(reference, complex)
    w = np.asarray(weights, float); m = np.asarray(phase_mask, bool)
    den = np.sum(w * abs(r)**2)
    y = np.unwrap(np.angle(r[m] * np.conj(c[m])))
    wm = w[m] * abs(r[m])**2
    X = np.column_stack((np.ones(m.sum()), 2*np.pi*f[m]))
    beta = np.linalg.lstsq(X * np.sqrt(wm)[:, None], y * np.sqrt(wm), rcond=None)[0]
    constant_phase = float(np.angle(np.sum(w * np.conj(c) * r)))
    aligned = [c, c*np.exp(1j*constant_phase), c*np.exp(1j*(beta[0]+2*np.pi*f*beta[1]))]
    rms_phase = []
    for a in (c[m], c[m]*np.exp(1j*constant_phase), c[m]*np.exp(1j*(beta[0]+2*np.pi*f[m]*beta[1]))):
        p = np.unwrap(np.angle(r[m] * np.conj(a)))
        rms_phase.append(float(np.sqrt(np.sum(wm*p*p)/np.sum(wm))))
    l2 = [float(np.sqrt(np.sum(w*abs(a-r)**2)/den)) for a in aligned]
    return {'raw_phase_rms': rms_phase[0], 'constant_phase_rms': rms_phase[1],
            'affine_phase_rms': rms_phase[2], 'raw_relative_l2': l2[0],
            'constant_relative_l2': l2[1], 'affine_relative_l2': l2[2],
            'fit_constant_phase': float(beta[0]), 'fit_time_shift': float(beta[1])}


def overlap_at(f, candidate, reference, weights, dt):
    w=np.asarray(weights,float); c=np.asarray(candidate,complex); r=np.asarray(reference,complex)
    cross=np.sum(w*np.conj(c)*r*np.exp(-2j*np.pi*np.asarray(f)*dt))
    return float(abs(cross)/np.sqrt(np.sum(w*abs(c)**2)*np.sum(w*abs(r)**2)))


def dense_max(f,c,r,w,bounds=(-200.,200.), n=4001):
    grid=np.linspace(*bounds,n); vals=np.empty(n)
    for i,dt in enumerate(grid): vals[i]=overlap_at(f,c,r,w,dt)
    i=int(np.argmax(vals)); lo=grid[max(0,i-1)]; hi=grid[min(n-1,i+1)]
    opt=minimize_scalar(lambda x:-overlap_at(f,c,r,w,x),bounds=(lo,hi),method='bounded',options={'xatol':1e-11})
    return {'dense_grid_points':n,'dense_grid_dt':float(grid[i]),'refined_dt':float(opt.x),
            'refined_overlap':float(-opt.fun),'dense_grid_overlap':float(vals[i]),
            'boundary':bool(i in (0,n-1)), 'success':bool(opt.success)}


def main(report_path):
    rp=Path(report_path); report=json.loads(rp.read_text()); cases=[]; max_metric=0.; max_ov=0.; coeff_diff=0.; required_gate_ok=True
    protocol=json.loads((rp.parent.parent/'protocol.json').read_text())
    expected_ids=protocol['primary_ids']; limits=protocol['phase_quality_diagnostic_limits']
    white_interior=psd_interior=True; dense=None
    actual_optimizer_bounds=True; holdout_splits=True; aggregate_gates=True
    lo, hi = protocol['overlap']['time_bounds_M']
    for ci,case in enumerate(report['cases']):
        z=np.load(case['arrays']['path'],allow_pickle=False); f=z['frequency_Mf']; r=z['primary']; w=z['quadrature_weights']; mask=z['phase_mask'].astype(bool)
        holdout_splits &= case['split'] == 'holdout'
        npz_hash=hashlib.sha256(Path(case['arrays']['path']).read_bytes()).hexdigest()
        numeric_finite=all(np.all(np.isfinite(z[k])) for k in z.files if np.issubdtype(z[k].dtype, np.number))
        weights_ok=bool(np.all(np.isfinite(w)) and np.all(w>0))
        retained=float(np.sum(w[mask]*abs(r[mask])**2)/np.sum(w*abs(r)**2))
        row={'case_id':case['case_id'],'bins':int(f.size),'finite_arrays':bool(numeric_finite),'weights_positive':weights_ok,'npz_hash_matches_report':npz_hash==case['arrays']['sha256'],'phase_mask_retained_reference_norm_fraction':retained,'controls':{},'lal':[]}
        for item in case['numerical_controls']:
            k=item['key']; got=metrics(f,z[k],r,w,mask); saved=item['metrics']; dif=max(abs(got[x]-saved[x]) for x in ('raw_phase_rms','constant_phase_rms','affine_phase_rms','raw_relative_l2','constant_relative_l2','affine_relative_l2')); max_metric=max(max_metric,dif)
            cd=max(abs(got['fit_constant_phase']-saved['affine_phase']),abs(got['fit_time_shift']-saved['time_shift'])); coeff_diff=max(coeff_diff,cd)
            lim=limits['preprocessing_phase_rms_rad' if k.startswith('preprocessing') else 'extraction_resolution_phase_rms_rad']; llim=limits['preprocessing_aligned_complex_l2' if k.startswith('preprocessing') else 'extraction_resolution_aligned_complex_l2']
            gate=(got['affine_phase_rms']<=lim and got['affine_relative_l2']<=llim); required_gate_ok &= (gate==bool(item['passed']))
            row['controls'][k]={'max_abs_metric_difference':float(dif),'coefficient_max_abs_difference':float(cd),'direct':got,'recomputed_gate':gate,'saved_phase_quality_pass':item['passed']}
        for lal in case['lal_comparisons']:
            M=int(lal['total_mass_msun']); c=z[f'lal_M{M}']; hz=z[f'frequency_Hz_M{M}']; psd=z[f'psd_M{M}']; keep=hz>=20
            ww=w[keep]/psd; ww=ww/ww.max(); saved=lal['metrics']; got=metrics(f,c,r,w,mask)
            dif=max(abs(got[x]-saved[x]) for x in ('raw_phase_rms','constant_phase_rms','affine_phase_rms','raw_relative_l2','constant_relative_l2','affine_relative_l2')); max_metric=max(max_metric,dif)
            wo=lal['unweighted_optimized']; po=lal['detector_optimized']; wd=w; pd=ww
            coeff_diff=max(coeff_diff, abs(got['fit_constant_phase']-saved['affine_phase']), abs(got['fit_time_shift']-saved['time_shift']))
            actual_optimizer_bounds &= all(lo < v['time_shift'] < hi for v in (wo, po))
            wod=overlap_at(f,c,r,wd,wo['time_shift']); pod=overlap_at(f[keep],c[keep],r[keep],pd,po['time_shift'])
            max_ov=max(max_ov,abs(wod-wo['overlap']),abs(pod-po['overlap']))
            white_interior &= not wo['boundary']; psd_interior &= not po['boundary']
            row['lal'].append({'total_mass_msun':M,'metrics_max_abs_difference':float(dif),'direct_metrics':got,'white_overlap_at_saved_shift':wod,'saved_white_overlap':wo['overlap'],'psd_overlap_at_saved_shift':pod,'saved_psd_overlap':po['overlap'],'white_interior':not wo['boundary'],'psd_interior':not po['boundary'],'psd_finite_positive':bool(np.all(np.isfinite(psd)) and np.all(psd>0))})
            if ci==0 and M==20: dense=dense_max(f[keep],c[keep],r[keep],pd)
        direct_required_pass=all(row['controls'][v['key']]['recomputed_gate'] for v in case['numerical_controls'] if v['required'])
        aggregate_gates &= direct_required_pass and direct_required_pass == bool(case['phase_quality_pass'])
        cases.append(row)
    checks={'cases':len(cases),'case_ids_match_protocol':[x['case_id'] for x in cases]==expected_ids,'all_holdouts_present':len(cases)==len(expected_ids),'all_saved_arrays_finite':all(x['finite_arrays'] for x in cases),'all_weights_positive':all(x['weights_positive'] for x in cases),'all_npz_hashes_match_report':all(x['npz_hash_matches_report'] for x in cases),'minimum_phase_mask_reference_norm_fraction':min(x['phase_mask_retained_reference_norm_fraction'] for x in cases),'max_direct_phase_l2_metric_difference':max_metric,'max_fit_coefficient_difference':coeff_diff,'all_required_gate_flags_match_direct_replay':required_gate_ok,'max_direct_overlap_difference':max_ov,'all_white_maxima_interior':white_interior,'all_psd_maxima_interior':psd_interior,'all_optimizer_times_in_bounds':all(-200<=x['direct_metrics']['fit_time_shift']<=200 for c in cases for x in c['lal']),'all_psds_finite_positive':all(x['psd_finite_positive'] for c in cases for x in c['lal']),'representative_dense_grid':dense,'dense_refinement_success':bool(dense is not None and np.isfinite(dense['refined_overlap'])),'dense_overlap_abs_difference_vs_saved':abs(dense['refined_overlap']-cases[0]['lal'][0]['saved_psd_overlap']) if dense else float('inf'),'new_calibration':report['new_calibration'],'runtime_promoted':report['runtime_promoted']}
    checks['status_conditions_ok']=all([checks['case_ids_match_protocol'],checks['all_holdouts_present'],checks['all_saved_arrays_finite'],checks['all_weights_positive'],checks['all_npz_hashes_match_report'],checks['all_required_gate_flags_match_direct_replay'],checks['max_direct_phase_l2_metric_difference']<=1e-10,checks['max_fit_coefficient_difference']<=1e-10,checks['all_white_maxima_interior'],checks['all_psd_maxima_interior'],checks['all_optimizer_times_in_bounds'],checks['all_psds_finite_positive'],checks['dense_refinement_success'],checks['dense_overlap_abs_difference_vs_saved']<=1e-10])
    checks['all_optimizer_times_in_bounds'] = bool(actual_optimizer_bounds)
    checks['all_holdout_splits_verified'] = bool(holdout_splits)
    checks['all_required_phase_gates_pass'] = bool(aggregate_gates)
    checks['dense_refinement_success'] = bool(dense and dense['success'] and not dense['boundary'])
    checks['status_conditions_ok'] = bool(checks['status_conditions_ok'] and actual_optimizer_bounds and holdout_splits and aggregate_gates and checks['dense_refinement_success'] and max_ov <= 1e-10)
    out={'schema':'nr33-phase-benchmark-independent-replay-v3','report_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'status':'PASS' if checks['status_conditions_ok'] else 'FAIL','checks':checks,'cases':cases}
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('report',nargs='?',default='docs/reviews/2026-09-15-independent-phase-benchmark/results/report.json'); ap.add_argument('--output'); a=ap.parse_args(); out=main(a.report); s=json.dumps(out,indent=2)+'\n'; Path(a.output).write_text(s) if a.output else print(s); print(json.dumps({'status':out['status'],'checks':out['checks']},indent=2)); raise SystemExit(0 if out['status']=='PASS' else 1)
