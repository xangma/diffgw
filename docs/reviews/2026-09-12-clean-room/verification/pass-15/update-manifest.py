"""Record completed pass15 results without changing acceptance criteria."""
import copy, datetime, hashlib, json, re
from pathlib import Path
out=Path(__file__).resolve().parent
root=out.parents[4]
path=root/'docs/model_manifest.json'
old_hash=hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads(path.read_text())
run=json.loads((out/'verification-run.json').read_text())
assert run['source_stable'] and run['runs'][0]['returncode']==0
match=re.search(r'(\d+) passed, (\d+) skipped in ([\d.]+)s',(out/'pytest.log').read_text())
passed,skipped,seconds=int(match[1]),int(match[2]),float(match[3])
reports={scope:json.loads((out/f'{scope}.json').read_text()) for scope in ('catalog','holdout')}
for name,model in m['models'].items():
 for scope,report in reports.items():
  v=model.get('validation',{}).get(scope)
  if v is None:continue
  cases=[r for r in report['results'] if r['case_id'] in v['case_ids']]
  assert len(cases)==len(v['case_ids'])
  v['cases_passed']=sum(r['status']=='PASSED' for r in cases)
  pol=[r[k]['complex_relative_l2'] for r in cases for k in ('plus','cross') if k in r]
  proj=[p['complex_relative_l2'] for r in cases for p in r.get('projections',{}).values()]
  v['max_polarization_complex_relative_l2']=max(pol,default=None)
  v['max_projection_complex_relative_l2']=max(proj,default=None)
m['models']['IMRPhenomXHM']['notes'] += ' Pass 15 adds partial-node quartics, conditional high-q node removal, and numerical stationary-point denominator checks with selected value-only fallback. Four-point A/A0 monotonicity is an explicit interpretation of an underspecified rule; numeric amplitude threshold units remain unresolved and inactive. Raw fits and acceptance gates are unchanged. Strict passing counts did not increase and several independent XHM/XPHM waveform errors worsened. Sparse-node selection and calibration metadata remain unresolved; full parity remains unachieved.'


v=copy.deepcopy(m['verification'])
v.pop('supplemental_explicit_nnlo',None)
v.update(evidence='docs/reviews/2026-09-12-clean-room/verification/pass-15/', unit_tests_passed=passed,unit_tests_skipped=skipped,unit_tests_seconds=seconds)
for scope,r in reports.items():
 v[f'{scope}_cases_passed']=r['num_passed'];v[f'{scope}_cases_total']=r['num_cases']
c=json.loads((out/'components-after.json').read_text())
v['supplemental_explicit_components']=dict(cases_passed=c['num_passed'],cases_total=c['num_cases'],mode_sum_controls_passed=sum(g['passed'] for g in c['mode_sum_controls']),evidence='docs/reviews/2026-09-12-clean-room/verification/pass-15/comparison.json',scope='Separate NNLO102/FinalSpin0 and MSA223/FinalSpin3 plus HM/XHM selected-mode diagnostics. XHM selects both signed modes and disables multibanding. Overlapping controls do not add frozen passing cases.')
m['verification']=v;m['verification/pass-15']=copy.deepcopy(v)
path.write_text(json.dumps(m,indent=2)+'\n')
(out/'post-verification-doc-update.json').write_text(json.dumps(dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),file='docs/model_manifest.json',sha256_before=old_hash,sha256_after=hashlib.sha256(path.read_bytes()).hexdigest(),purpose='Record final verification results and implementation limits only.'),indent=2)+'\n')
print('Manifest updated:',passed,skipped,seconds)
