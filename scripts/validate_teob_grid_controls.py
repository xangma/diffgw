#!/usr/bin/env python3
"""Validate recorded frozen-grid controls without rerunning an oracle."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/reviews/2026-09-15-teob-grid-nqc'
OLD=ROOT/'docs/reviews/2026-09-15-teob-fd-attribution'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    derivative=read(OUT/'derivative-controls.json')
    assert derivative['status']=='finished' and not derivative['source_changes']
    runtime= {p:h for p,h in derivative['source_hashes_before'].items() if p.startswith('src/')}
    assert all(sha(ROOT/p)==h for p,h in runtime.items())
    old_manifest=read(OLD/'artifact-manifest.json')
    historical_changes=[p for p,h in old_manifest.items() if sha(Path(p))!=h]
    assert historical_changes==[str(ROOT/'scripts/teob_fd_matching_controls.py')]
    assert sha(OUT/'before/teob_fd_matching_controls.py')==old_manifest[historical_changes[0]]
    rows={}
    for name,case in derivative['cases'].items():
        assert sha(OLD/name/'continuous.npz')==case['input_sha256']
        for variant,detail in case['variants'].items():
            assert sha(OUT/name/f'{variant}.npz')==detail['output_sha256']
            assert detail['acceptance']['status']=='FAILED'
            if variant=='previous_replay':assert detail['previous_replay_bitwise']
            else:assert detail['td_grid_identical'] and detail['attachment_identical']
        nominal=np.load(OUT/name/'nominal.npz',allow_pickle=False)
        replay_path=OUT/'validated-replay'/name/'matching-grid.npz'
        replay=np.load(replay_path,allow_pickle=False)
        assert set(nominal.files)==set(replay.files)
        assert all(np.array_equal(nominal[k],replay[k]) for k in nominal.files)
        replay_report=read(replay_path.parent/'matching-controls.json')
        assert not replay_report['source_changes']
        assert replay_report['fixed_step_M']==.1
        assert replay_report['variants']['grid']['cache_sha256']==sha(replay_path)
        output=read(OUT/name/'output-resampling.json')
        assert output['status']=='finished' and not output['source_changes']
        assert output['nominal_sha256']==sha(OUT/name/'nominal.npz')
        for variant,detail in output['variants'].items():
            assert sha(OUT/name/f'output-{variant}.npz')==detail['output_sha256']
            assert detail['post_attachment_mode_bitwise']
            assert detail['acceptance']['status']=='FAILED'
        assert output['variants']['baseline']['nominal_bitwise']
        rows[name]={'derivative_outputs_verified':len(case['variants']),
                    'output_resampling_outputs_verified':len(output['variants']),
                    'corrected_tool_replay_bitwise':True}
    localizations={}
    for name in ('catalog','bbh_reference'):
        loc=read(OUT/f'localization-{name}.json')
        controls=read(OLD/name/'controls.json')
        nominal=np.load(OUT/name/'nominal.npz',allow_pickle=False)
        public=np.load(controls['inputs']['public_td_npz'],allow_pickle=False)
        ref=public['exact_start_raw_h_plus']-1j*public['exact_start_raw_h_cross']
        value=nominal['h_plus_td']-1j*nominal['h_cross_td']
        measured=float(np.linalg.norm(value-ref)/np.linalg.norm(ref))
        assert measured==loc['global_td']['hp_minus_i_hc']['relative_l2']
        assert abs(sum(s['share_of_total_squared_residual'] or 0 for s in loc['segments'])-1)<1e-12
        assert sum(s['samples'] for s in loc['segments'])==len(value)
        assert loc['input_hashes']['derivative_controls_json']==sha(OUT/'derivative-controls.json')
        for item in loc['fixture_verification'].values():
            if isinstance(item,dict) and 'hash_match' in item:assert item['hash_match']
        arr=np.load(OUT/f'localization-{name}.npz',allow_pickle=False)
        cum=arr['normalized_cumulative_residual_energy']
        localizations[name]={'independent_complex_l2_verified':True,
            'cumulative_residual_energy_indices':{str(q):int(np.searchsorted(cum,q)) for q in (.1,.5,.9,.99)}}
    report={'status':'passed','runtime_files_unchanged':len(runtime),
            'historical_artifacts_unchanged':len(old_manifest)-len(historical_changes),
            'historical_tool_updated_with_original_saved':historical_changes,'cases':rows,
            'localizations':localizations,'validation_script_sha256':sha(Path(__file__))}
    (OUT/'artifact-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
