"""Record reviewed pass-13 changes and completed fixed-inventory verification."""
import datetime
import difflib
import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPORT = OUT.parents[1]
ROOT = OUT.parents[4]
BEFORE = Path('/tmp/torchwave-pass13-before')


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


run = read('verification-run.json')
comparison = read('comparison.json')
assert run['source_stable'] and run['runs'][0]['returncode'] == 0
passed, skipped, seconds = re.search(
    r'(\d+) passed, (\d+) skipped in ([\d.]+)s', (OUT / 'pytest.log').read_text()
).groups()
hash_count = len(run['source_sha256_after'])
reports = {s: read(f'{s}.json') for s in ('catalog', 'holdout')}
components = read('components-after.json')

changed = []
patch = []
for area in ('src', 'tests', 'scripts'):
    for path in sorted((ROOT / area).rglob('*.py')):
        relative = path.relative_to(ROOT)
        old_path = BEFORE / relative
        old = old_path.read_text() if old_path.exists() else ''
        new = path.read_text()
        if old != new:
            changed.append(str(relative))
            patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                         fromfile=f'before/{relative}', tofile=f'after/{relative}'))
(OUT / 'changes-from-pass12.patch').write_text(''.join(patch))
assert set(changed) == {
    'src/torchwave/approximants/imrphenom_xhm.py',
    'tests/test_pass12_higher_modes.py',
    'tests/test_pass13_xhm_amplitude.py',
    'tests/test_pass13_xhm_phase.py',
    'tests/test_clean_higher_modes.py',
}
save('integration-review.json', dict(
    changed_files=changed,
    runtime_review='Parent checked Eqs4.5/4.6/5.3/6.2, TablesIV/VI/VII and Fig8 directly. Retained constant-A0 inner targets, separate derivative-removal value cubic for poles versus inner-value-removal Hermite cubic for parameter vetoes, row-global linear fallback, and default nonmixing phase nodes [1,2,3,4,6]. The second Luna independently reviewed TableVI. Earlier dirty work and acceptance harness remain unchanged.',
    test_review='Both 21 nodes are checked against normalized physical strain with zero absolute tolerance. A captured production 6x6 solve proves the constructed quintic triggers the existing interval detector while its four-value cubic stays positive; both fallback inner values and finite/nonzero frequency gradients are checked before forcing the distinct parameter veto. Phase solve node/RHS capture, p6 strain sensitivity, p5 exclusion for all three nonmixing modes and phase-sensitive mass/spin gradients are checked. Parent made the independent detector grid explicitly FP64.',
    existing_test_change='The mixed-batch parameter-veto test uses 21 native physical inputs because corrected normalization activates an unrelated pole fallback in its former 32 control. Zero-tolerance veto invariance, sensitive non-veto control, disabling-veto control and finite/nonzero gradients remain. The 32 join test independently proves parameter veto is inactive and pole fallback active, preserving C0 and phase checks. A separate complete-waveform control uses inner fit values derived from physical endpoint Hermite conditions, proves its production quintic clears the detector and verifies C0/C1 amplitude at both joins. Parent replaced the initial polynomial-only control; focused integration passed 57 tests.',
    verification=dict(passed=int(passed), skipped=int(skipped), seconds=float(seconds),
                      source_hashes_stable=hash_count),
    earlier_candidate='normalization-and-nodes/ preserves the initial verified candidate before the independent TableVI fallback correction. It is historical, not the final result.',
))
audit = read('source-audit.json')
audit['integration'] += ' TableVI pole fallback retains all four values and removes endpoint derivatives; parameter veto removes inner values. Cubic/linear selection is global per intrinsic row. Independent Luna review is recorded in fallback-independent-review.md.'
save('source-audit.json', audit)

rows = []
for scope in ('catalog', 'holdout'):
    for name, pair in comparison[scope]['changed_case_metrics_from_pass12'].items():
        b, a = pair['before'], pair['after']
        rows.append(f"| {name} | {b['max_polarization_complex_relative_l2']:.8g} → {a['max_polarization_complex_relative_l2']:.8g} | {b['max_polarization_magnitude_relative_l2']:.8g} → {a['max_polarization_magnitude_relative_l2']:.8g} |")

text = f'''# Pass 13: XHM normalization, fallback and phase-node corrections

Completed 13 September 2026 with two fresh-context **GPT-5.6 Luna** subagents
and parent integration/verification. **Full LAL parity remains unfinished.**
The strict catalog passes **{reports['catalog']['num_passed']}/{reports['catalog']['num_cases']}** and frozen holdouts **{reports['holdout']['num_passed']}/{reports['holdout']['num_cases']}**.
No new strict case passes, and all four full XHM/XPHM complex errors worsen.
This pass corrects source-defined behavior without demonstrating full-waveform
parity progress.

## Retained changes

- Intermediate amplitude fit targets use the paper's frequency-independent
  `A0`: inverse-polynomial inner values are `1/(A/A0)`. Conversion to the
  native frequency-dependent leading-amplitude ratio occurs once at evaluation.
- Parameter vetoes remove inner values and retain endpoint derivatives.
  Denominator failure instead removes endpoint derivatives and retains four
  values. Each fallback selects one polynomial for the complete interval;
  a failing cubic selects the endpoint linear reconstruction globally.
- Nonmixing phase reconstruction uses the published default nodes
  `[1,2,3,4,6]` and their inherited fit tables, replacing `[1,2,3,4,5]`.

The [patch](verification/pass-13/changes-from-pass12.patch) changes one runtime
file, two new test files and two existing tests. Earlier dirty work is
preserved. Calibration tables, acceptance inventories, grids and thresholds
are unchanged. See the [amplitude derivation](pass-13-amplitude.md),
[phase review](pass-13-phase-metadata.md),
[independent fallback review](verification/pass-13/fallback-independent-review.md)
and [integration review](verification/pass-13/integration-review.json).

## Verification

| Check | Result |
| --- | --- |
| Full ordinary suite | **{int(passed):,} passed, {skipped} skipped, {seconds} seconds** |
| Frozen strict catalog | **{reports['catalog']['num_passed']}/{reports['catalog']['num_cases']} passed** |
| Frozen strict holdouts | **{reports['holdout']['num_passed']}/{reports['holdout']['num_cases']} passed** |
| Separate explicit component checks | **{components['num_passed']}/{components['num_cases']} passed** |
| Mode-sum closure controls | **{sum(c['passed'] for c in components['mode_sum_controls'])}/4 passed** |

The catalog still contains nine numerical failures and one unresolved complete
reference. Component successes overlap carrier checks and add no frozen parity
success. Ordinary tests establish mathematical/API behavior, not LAL agreement.
Six skips require CUDA and two require generation-only gmpy2; MPS tests pass.

Changed frozen maximum polarization relative L2 errors (before → after):

| Case | Complex strain | Magnitude only |
| --- | ---: | ---: |
{chr(10).join(rows)}

All four changed full-waveform complex errors worsen. Some isolated component
amplitude errors improve; this does not establish full-waveform progress.
All other frozen case metrics are unchanged. The complete measurements and
unchanged-inventory assertions are in the
[comparison](verification/pass-13/comparison.json). Improvements and regressions
are reported together; no residuals were supplied to the implementers or used
to fit coefficients. The earlier normalization/node-only candidate is preserved
under [normalization-and-nodes](verification/pass-13/normalization-and-nodes/README.md).
Subsequent source review identified the distinct Table VI fallback rule and
both Luna reviews confirmed it before final verification.

The first fallback verification found two older unconditional C1 amplitude
assertions. The published pole branch removes those derivative conditions.
Its default test now proves the pole branch is active and preserves C0 and
phase checks. A separate complete-waveform control sets inner fit values from
the actual endpoint Hermite polynomial, proves the production quintic clears
the detector, and verifies C0/C1 at both amplitude joins. No tolerances were
relaxed. The intermediate failed run is preserved under
[fallback-before-join-test-correction](verification/pass-13/fallback-before-join-test-correction/README.md).
The final focused integration passed **57 tests**.

The final verifier ran on **{run['host']}**, cwd `{run['cwd']}`, driver PID
**{run['pid']}**, from `{run['started_utc']}` to `{run['ended_utc']}` using
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` and the
[driver](verification/pass-13/verify.py). All children finished; exits are
{('/'.join(str(r['returncode']) for r in run['runs']))} for ordinary/catalog/holdout/component checks.
Exact commands, child PIDs and logs are in the
[run record](verification/pass-13/verification-run.json). All **{hash_count}**
runtime/test/script/manifest hashes remained stable during verification.
The later manifest edit records results separately. `git diff --check` passes.

## Remaining limits

The pole detector still uses the inherited 11-point/5%-of-endpoint heuristic;
this is not an exact root search. Ordered absolute-amplitude vetoes remain
incomplete. Sparse-calibration phase-node selection, independent p6 table
provenance and author phase-fit preprocessing remain unresolved. No speculative
relative phase-target conversion was introduced. These gaps and the existing
MSA, EOB, ROM and Kerr waveform work prevent a credible completion percentage.
See the [remaining-work assessment](pass-12-remaining-work.md).

Implementers used primary papers, author mathematical assets and inherited
native code. LAL/PyCBC implementation source and binary residuals were excluded;
parent public-binary verification was separate. This instruction boundary does
not certify inherited provenance or provide OS-enforced isolation. Existing
tables remain in use; the deferred NR job was not restarted or scheduled.
No commit or pull request was created.
'''
(REPORT / 'pass-13-summary.md').write_text(text)

path = REPORT / 'summary.md'
text = path.read_text()
start = text.index('figures below describe')
end = text.index('Pass-7 Kerr', start)
text = text[:start] + '''figures below describe the completed [pass-13 verification](pass-13-summary.md):
XHM constant-A0 intermediate amplitude targets, source-defined polynomial
fallbacks and the default nonmixing phase-node subset. **Pass 13 adds no strict
passing case; all four full XHM/XPHM complex errors worsen.**
''' + text[end:]
text = text.replace('passes 9--12', 'passes 9--13')
text = text.replace('**1,112 tests, with eight skipped, in\n76.14 s**',
                    f'**{int(passed):,} tests, with eight skipped, in\n{seconds} s**')
text = text.replace('All 171 source', f'All {hash_count} source')
text = text.replace('Unconditional parameter-region amplitude vetoes now select the cubic boundary fit; absolute-amplitude and phase conventions remain unresolved.',
                    'Intermediate nodes now use constant-A0 targets. Parameter vetoes retain boundary derivatives; pole fallbacks retain four values. Phase defaults use nodes [1,2,3,4,6]. Absolute-amplitude vetoes, sparse node selection and fit conventions remain unresolved.')
text = text.replace('[pass-12 vetoes](pass-12-higher-modes.md) |',
                    '[pass-12 vetoes](pass-12-higher-modes.md), [pass-13 corrections](pass-13-summary.md) |')
for report in reports.values():
    for r in report['results']:
        if 'plus' not in r:
            continue
        pol = max(r[p]['complex_relative_l2'] for p in ('plus', 'cross'))
        projection = max(p['complex_relative_l2'] for p in r['projections'].values())
        row = f"| {r['case_id']} | {pol:.8g} | {projection:.8g} | {r['status']} |"
        text = re.sub(r'^\| ' + re.escape(r['case_id']) + r' \|.*$', row, text, flags=re.M)
needle = 'The independent [Kerr component measurements]'
paragraph = '''Pass 13 corrects XHM intermediate amplitude normalization, separates the
published parameter and pole fallbacks, and applies the default phase node f6.
Strict passing counts remain unchanged; all four full XHM/XPHM complex errors
worsen, despite some component amplitude improvements. Other frozen metrics
are unchanged. The separate component probe
still passes 3/34 and all four mode-sum controls pass. See the
[pass-13 report](pass-13-summary.md) for before/after complex and magnitude
errors, source evidence, and remaining fit-convention limits.

'''
text = text.replace(needle, paragraph + needle)
path.write_text(text)
print('Updated pass-13 report, main summary, integration review and patch.')
