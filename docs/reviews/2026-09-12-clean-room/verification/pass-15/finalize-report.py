"""Record reviewed pass 15 implementation and completed independent verification."""
import difflib, json, re
from pathlib import Path
OUT = Path(__file__).resolve().parent
REPORT = OUT.parents[1]
ROOT = OUT.parents[4]
BEFORE = Path('/tmp/torchwave-pass15-before')
def read(name): return json.loads((OUT/name).read_text())
run=read('verification-run.json'); comparison=read('comparison.json')
assert run['source_stable'] and run['runs'][0]['returncode']==0
passed, skipped, seconds=re.search(r'(\d+) passed, (\d+) skipped in ([\d.]+)s',(OUT/'pytest.log').read_text()).groups()
reports={scope:read(f'{scope}.json') for scope in ('catalog','holdout')}
components=read('components-after.json')
changed=[]; patch=[]
for path in sorted([*ROOT.glob('src/**/*.py'),*ROOT.glob('tests/**/*.py'),*ROOT.glob('scripts/**/*.py'),*ROOT.glob('tests/data/*.json')]):
 relative=path.relative_to(ROOT); old_path=BEFORE/relative
 old=old_path.read_text() if old_path.exists() else ''; new=path.read_text()
 if old != new:
  changed.append(str(relative))
  patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=f'before/{relative}',tofile=f'after/{relative}'))
assert [p for p in changed if p.startswith('src/')] == ['src/torchwave/approximants/imrphenom_xhm.py','src/torchwave/approximants/polynomial_interval.py']
assert not any(p.startswith('scripts/') or p.startswith('tests/parity_') or p.startswith('tests/data/') for p in changed)
(OUT/'changes-from-pass14.patch').write_text(''.join(patch))
(OUT/'integration-review.json').write_text(json.dumps(dict(
 changed_files=changed,
 runtime_review='Parent reviewed the incremental runtime diff. Monotonicity compares all four A/A0 values (documented interpretation), high-q veto removes each remaining node. Partial-node quartics retain both boundary derivatives and the selected value. Pole fallback removes both derivatives while retaining surviving values; linear is selected only when both the active and corresponding value-only polynomial fail. Parent replaced provisional swapped/ungated sampled branches with selected coefficient arrays and two interval checks. No raw calibration coefficients changed.',
 numerical_review='Derivative recursion, affine interval transformation, amplitude-scaled tolerance, broadcast and detached decisions reviewed. Parent compared 600 seeded degree0..5 polynomials against independent NumPy derivative-root extrema with no mismatches. Numerical guard, not exact certification: near-roundoff minima, clustered critical roots and extreme overflow/underflow remain floating-point limitations.',
 test_review='Production checks cover both partial-node directions, common endpoint/interior amplitude units, heterogeneous gradients and inactive fallback isolation. Two older collocation/fallback tests now explicitly retain nodes unless parameter-vetoed so they test their original reconstruction contract independently of new monotonicity selection. Genuine negative-denominator assertion strengthened from 5% threshold to zero. Frozen parity inventories, cases, grids and tolerances unchanged.',
 focused_tests_passed=71,focused_tests_skipped=1,
 ordinary_tests_passed=int(passed),ordinary_tests_skipped=int(skipped),
 input_hashes_stable=len(run['source_sha256_after']),
 source_boundary='Three fresh-context GPT-5.6 Luna subagents used primary-paper mathematics and native source/tests. Excluded LAL/PyCBC source, translated ports, notebooks, source-derived prior spec, all verifier artifacts and waveform binary residuals. Parent-only binary validation, no coefficient tuning. Instruction boundary, not OS isolation; inherited provenance uncertified.'
),indent=2)+'\n')
rows=[]
for scope in ('catalog','holdout'):
 for name,pair in comparison[scope]['changed_case_metrics_from_pass14'].items():
  b,a=pair['before'],pair['after']
  rows.append(f"| {name} | {b['max_polarization_complex_relative_l2']:.8g} → {a['max_polarization_complex_relative_l2']:.8g} | {b['max_polarization_magnitude_relative_l2']:.8g} → {a['max_polarization_magnitude_relative_l2']:.8g} |")
metrics_text=('| Changed frozen case | Complex strain | Magnitude only |\n| --- | ---: | ---: |\n'+'\n'.join(rows)) if rows else 'Frozen catalog and holdout metrics are unchanged from pass 14.'
old_counts={s:json.loads((OUT.parent/'pass-14'/f'{s}.json').read_text())['num_passed'] for s in reports}
no_gain=all(reports[s]['num_passed']==old_counts[s] for s in reports)
status='No additional strict case passes; full LAL parity remains unfinished.' if no_gain else 'Full LAL parity remains unfinished; completed strict results are below.'
(REPORT/'pass-15-summary.md').write_text(f'''# Pass 15: ordered amplitude selection and interval denominator checks

Completed with three fresh-context **GPT-5.6 Luna** subagents, parent integration,
and separate parent-only public-binary verification. **{status}**

XHM now removes the higher-frequency inner amplitude node when the four
collocation amplitudes are nonmonotonic in common `A/A0` units. The four-value
sequence is an explicitly documented interpretation of an underspecified
paper rule, not a uniquely established prescription. The `q > 40`, `chi1 > 0.9`
rows remove each remaining inner node. Removing one node retains the other
and both boundary derivatives through a quartic inverse polynomial.

The old eleven-sample, five-percent denominator heuristic is replaced by a
native Torch numerical check of endpoints and stationary points throughout
the interval. Derivative-root isolation uses affine interval normalization
and batched bisection. The selected value-only fallback preserves surviving
nodes; a still-invalid fallback reduces to the endpoint line. Only the active
branch can trigger a fallback. Discrete decisions are detached while selected
polynomial gradients remain available.

The [independent source review](pass-15-veto-independent-review.md) found that
Table VI does not resolve raw `A` versus `A/A0` units for the `0.2`/`0.01`
thresholds. They remain inactive, including the `(3,2)` special derivative
veto. Both boundary derivatives are removed together for pole fallback; the
paper does not specify a one-at-a-time removal order. See the
[veto implementation](pass-15-vetoes.md) and [numerical method](pass-15-root-detection.md).

## Verification

| Check | Result |
| --- | --- |
| Focused reconstruction and new tests | 71 passed, 1 skipped |
| Independent parent polynomial extrema comparison | 600/600 matched |
| Full ordinary suite | **{int(passed):,} passed, {skipped} skipped, {seconds} seconds** |
| Frozen strict catalog | **{reports['catalog']['num_passed']}/{reports['catalog']['num_cases']} passed** |
| Frozen strict holdouts | **{reports['holdout']['num_passed']}/{reports['holdout']['num_cases']} passed** |
| Separate explicit components | **{components['num_passed']}/{components['num_cases']} passed** |
| Mode-sum closure controls | **{sum(c['passed'] for c in components['mode_sum_controls'])}/4 passed** |

Inventories, grids and acceptance gates are unchanged. The full ordinary suite
includes a new CUDA-conditional detector test; seven skips require CUDA and
two require generation-only gmpy2. MPS tests pass. Two older unit tests now
explicitly retain nodes to isolate their original normalization/fallback
contract; no external parity test was altered.

{metrics_text}

The [comparison](verification/pass-15/comparison.json) retains complete before
and after component diagnostics, including regressions. These overlapping
controls add no frozen passing cases. No waveform binary residuals were given
to implementers or used for calibration fitting.

The verifier ran on **{run['host']}**, cwd `{run['cwd']}`, driver PID
**{run['pid']}**, from `{run['started_utc']}` to `{run['ended_utc']}` with
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` and the
[driver](verification/pass-15/verify.py). All children finished. Exact commands,
PIDs, logs and exits are in the [run record](verification/pass-15/verification-run.json).
All **{len(run['source_sha256_after'])}** runtime/test/script/fixture/manifest hashes
were stable during verification; the later manifest change only records results.
See the [incremental patch](verification/pass-15/changes-from-pass14.patch) and
[parent review](verification/pass-15/integration-review.json).

## Remaining work and limits

Resolve threshold normalization and the monotonicity/fallback interpretations
from additional independent primary material; finish sparse phase-node
selection and calibration-target metadata. Broader MSA, EOB, ROM and complete
Kerr waveform work remains in the [assessment](pass-12-remaining-work.md).
The interval helper is a floating-point guard, not an algebraic certificate:
near-roundoff minima, clustered critical roots, and extreme overflow/underflow
limit classification. The random comparison excludes minima within `1e-9`
of zero; focused tests separately cover tangencies and narrow negative dips.

Implementers used primary papers and native TorchWave files. LAL/PyCBC source,
translated ports, generated notebooks, the earlier source-derived specification,
and verification artifacts were excluded. Primary source:
[García-Quirós et al., arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2).
No new source assets or runtime oracle dependencies were added. Separation is
an instruction boundary, not OS isolation; inherited provenance is uncertified.
Existing calibration tables remain in use. The deferred NR job was not restarted
or scheduled. No commit or pull request was created.
''')
p=REPORT/'summary.md';s=p.read_text();a=s.index('[Pass 14]');b=s.index('Pass-7 Kerr',a)
s=s[:a]+f'''[Pass 15](pass-15-summary.md) adds ordered partial-node amplitude reconstruction
and numerical interval denominator checks. **{status}**
The figures below are the completed pass-15 measurements. Numeric amplitude
threshold units and the precise monotonicity prescription remain unresolved.
'''+s[b:]
s=s.replace('passes 9--14','passes 9--15')
a=s.index('The complete ordinary suite passed');b=s.index('The strict catalog passes',a)
s=s[:a]+f'''The complete ordinary suite passed **{int(passed):,} tests, with {skipped} skipped, in
{seconds} s**. Seven require CUDA and two require generation-only gmpy2; MPS tests
passed. All {len(run['source_sha256_after'])} runtime, test, script, fixture and manifest hashes remained
stable during verification. The later manifest update only records results.
'''+s[b:]
s=s.replace('pole fallbacks retain four values.', 'pole fallbacks retain surviving values. Partial-node quartics, conditional high-q vetoes and numerical interval denominator checks are active; four-point monotonicity is a documented interpretation.')
s=s.replace('[pass-14 sign convention](pass-14-summary.md) |','[pass-14 sign convention](pass-14-summary.md), [pass-15 selection](pass-15-summary.md) |')
for report in reports.values():
 for result in report['results']:
  if 'plus' not in result: continue
  pol=max(result[p]['complex_relative_l2'] for p in ('plus','cross'))
  projection=max(p['complex_relative_l2'] for p in result['projections'].values())
  row=f"| {result['case_id']} | {pol:.8g} | {projection:.8g} | {result['status']} |"
  s=re.sub(r'^\| '+re.escape(result['case_id'])+r' \|.*$',row,s,flags=re.M)
a=s.index('## Reproduction and provenance');head,repro=s[:a],s[a:]
repro=repro.replace('verification/pass-14/','verification/pass-15/').replace('rerun in pass 14','rerun in pass 15').replace('[pass-14 driver]','[pass-15 driver]').replace('[pass-14 verification]','[pass-15 verification]')
p.write_text(head+repro)
print('Reports updated',passed,skipped,len(changed))
