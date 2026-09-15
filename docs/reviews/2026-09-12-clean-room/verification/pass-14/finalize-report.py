"""Record reviewed Pass 14 changes and completed fixed-inventory verification."""
import difflib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
REPORT = OUT.parents[1]
ROOT = OUT.parents[4]
BEFORE = Path('/tmp/torchwave-pass14-before')


def read(name):
    return json.loads((OUT / name).read_text())


run = read('verification-run.json')
comparison = read('comparison.json')
assert run['source_stable'] and run['runs'][0]['returncode'] == 0
passed, skipped, seconds = re.search(
    r'(\d+) passed, (\d+) skipped in ([\d.]+)s', (OUT / 'pytest.log').read_text()
).groups()
reports = {s: read(f'{s}.json') for s in ('catalog', 'holdout')}
components = read('components-after.json')
changed, patch = [], []
paths = sorted([*ROOT.glob('src/**/*.py'), *ROOT.glob('tests/**/*.py'),
                *ROOT.glob('scripts/**/*.py'), *ROOT.glob('tests/data/*.json')])
for path in paths:
    relative = path.relative_to(ROOT)
    old_path = BEFORE / relative
    old = old_path.read_text() if old_path.exists() else ''
    new = path.read_text()
    if old != new:
        changed.append(str(relative))
        patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                     fromfile=f'before/{relative}', tofile=f'after/{relative}'))
assert [p for p in changed if p.startswith('src/')] == [
    'src/torchwave/approximants/imrphenom_xhm.py'
]
assert not any(p.startswith('scripts/') or p.startswith('tests/parity_') for p in changed)
(OUT / 'changes-from-pass13.patch').write_text(''.join(patch))
(OUT / 'integration-review.json').write_text(json.dumps(dict(
    changed_files=changed,
    runtime_review='Parent directly checked author Sec. V.A.2 sign-folding prescription and reviewed the two consumer-only odd-m abs blocks. Raw fit expressions and frozen verifier unchanged. Earlier dirty work preserved.',
    test_review='Production inspiral matmul and intermediate 6x6 solve targets captured for genuinely negative raw 21 fits. Assertions use zero absolute tolerance; waveform gradients finite and nonzero away from zero; even-m sign handling control retained. Parent replaced source-cache-only amplitude checks with native-vs-independent-fixture checks. Three phase and eight amplitude expressions match normalized author arithmetic ASTs. Independent fixtures regenerate byte-identically; all asset hashes checked. One amplitude hash transcription corrected against a fresh exact primary-source download.',
    focused_tests_passed=6,
    ordinary_tests_passed=int(passed), ordinary_tests_skipped=int(skipped),
    input_hashes_stable=len(run['source_sha256_after']),
    source_boundary='Fresh-context GPT-5.6 Luna implementers excluded LAL/PyCBC sources, translated ports, generation notebooks, verification artifacts and binary residuals. Parent-only public-binary checks do not tune coefficients. Instruction boundary, not OS isolation; inherited provenance uncertified.'
), indent=2) + '\n')

rows = []
for scope in ('catalog', 'holdout'):
    for name, pair in comparison[scope]['changed_case_metrics_from_pass13'].items():
        b, a = pair['before'], pair['after']
        rows.append(f"| {name} | {b['max_polarization_complex_relative_l2']:.8g} → {a['max_polarization_complex_relative_l2']:.8g} | {b['max_polarization_magnitude_relative_l2']:.8g} → {a['max_polarization_magnitude_relative_l2']:.8g} |")
metrics_text = ('| Changed frozen case | Complex strain | Magnitude only |\n'
                '| --- | ---: | ---: |\n' + '\n'.join(rows)) if rows else (
                    'All frozen catalog and holdout case metrics are unchanged from pass 13.')
assert reports['catalog']['num_passed'] == 11 and reports['holdout']['num_passed'] == 4

(REPORT / 'pass-14-summary.md').write_text(f'''# Pass 14: odd-mode amplitude correction and independent fit checks

Completed 13 September 2026 with three fresh-context **GPT-5.6 Luna** subagents
and parent review/verification. The odd-mode inspiral and intermediate amplitude
consumers now take the absolute value of signed calibration fits, as required
by the paper. Previously, negative intermediate values were clamped almost to
zero before inversion. Raw fit coefficients are unchanged.

The [source audit](pass-14-amplitude-assets.md) exposed negative raw `(2,1)`
fits at a chosen equal-mass, unequal-spin point. The
[runtime correction](pass-14-odd-amplitude.md) includes production-path target
checks and finite, nonzero gradients. Independent numerical fixtures also
check eight intermediate amplitude fits and three nonmixing p6 phase fits;
normal tests need no downloaded source cache. Parent review verifies their
arithmetic correspondence and reproducible generation. Matching expressions
does not establish production candidate selection or all calibration preprocessing.

**No additional strict case passes; full LAL parity remains unfinished.**
This is a source-supported behavior correction, not a full-waveform parity result.

## Verification

| Check | Result |
| --- | --- |
| Full ordinary suite | **{int(passed):,} passed, {skipped} skipped, {seconds} seconds** |
| Frozen strict catalog | **{reports['catalog']['num_passed']}/{reports['catalog']['num_cases']} passed** |
| Frozen strict holdouts | **{reports['holdout']['num_passed']}/{reports['holdout']['num_cases']} passed** |
| Separate explicit components | **{components['num_passed']}/{components['num_cases']} passed** |
| Mode-sum closure controls | **{sum(c['passed'] for c in components['mode_sum_controls'])}/4 passed** |

All suites were rerun after integration. The pass-13 measurements are used only
as the before baseline. Inventories, grids and gates are unchanged. Six skips
require CUDA and two require generation-only gmpy2; MPS tests pass. The catalog
still contains nine numerical failures and one unresolved complete reference.

{metrics_text}

The [comparison](verification/pass-14/comparison.json) retains complete before
and after component metrics. These overlapping controls add no frozen passing
case. No binary residuals were supplied to implementers or used to fit coefficients.

The verifier ran on **{run['host']}**, cwd `{run['cwd']}`, driver PID
**{run['pid']}**, from `{run['started_utc']}` to `{run['ended_utc']}` using
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python` and the
[driver](verification/pass-14/verify.py). All children finished; exact commands,
PIDs, logs and exit codes are in the [run record](verification/pass-14/verification-run.json).
All **{len(run['source_sha256_after'])}** runtime/test/script/fixture/manifest hashes
remained stable during verification. The later manifest edit only records results.
See the [patch](verification/pass-14/changes-from-pass13.patch) and
[parent integration review](verification/pass-14/integration-review.json).

## Remaining work and source boundary

Complete ordered amplitude vetoes, replace the inherited 11-point denominator
heuristic with complete interval root detection, and resolve sparse phase-node
selection and calibration-target metadata. The [phase audit](pass-14-phase-assets.md)
resolves p6 expression correspondence; repeating that lookup is unnecessary.
MSA, EOB, ROM and complete Kerr waveform work also remains, as described in the
[remaining-work assessment](pass-12-remaining-work.md).

Implementers used primary papers, author mathematical `.m` assets and native
TorchWave files. LAL/PyCBC source, translated ports, generation notebooks and
verifier outputs were excluded. Primary assets came from
[García-Quirós et al., arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2).
Complete downloaded records are ignored local review copies; tracked records
retain URLs, hashes and independently evaluated numerical fixtures. This does
not assert new redistribution rights or certify inherited provenance. The
clean-room separation is an instruction boundary, not OS isolation.

Existing calibration tables remain in use; the deferred NR job was not restarted
or scheduled. No commit or pull request was created.
''')

path = REPORT / 'summary.md'
text = path.read_text()
start, end = text.index('[Pass 14]'), text.index('Pass-7 Kerr')
text = text[:start] + '''[Pass 14](pass-14-summary.md) corrects signed odd-mode amplitude targets and
independently checks author expressions for eight intermediate amplitude fits
and three nonmixing p6 phase fits. **No additional strict case passes.**
The figures below are the completed pass-14 measurements. Expression matches
do not establish full calibration preprocessing or production candidate selection.
''' + text[end:]
start, end = text.index('The complete ordinary suite passed'), text.index('The strict catalog passes')
text = text[:start] + f'''The complete ordinary suite passed **{int(passed):,} tests, with {skipped} skipped, in
{seconds} s**. Six require CUDA and two require generation-only gmpy2; MPS tests
passed. All {len(run['source_sha256_after'])} runtime, test, script, fixture and manifest hashes remained
stable during verification. The later manifest update only records results.
''' + text[end:]
text = text.replace('Phase defaults use nodes [1,2,3,4,6].',
                    'Phase defaults use nodes [1,2,3,4,6]. Odd-m collocation targets now use absolute fitted amplitudes.')
text = text.replace('[pass-13 corrections](pass-13-summary.md) |',
                    '[pass-13 corrections](pass-13-summary.md), [pass-14 sign convention](pass-14-summary.md) |')
needle = 'The independent [Kerr component measurements]'
text = text.replace(needle, '''Pass 14 applies the source-defined absolute-value convention to odd-m amplitude
collocation targets and adds independent author-expression fixtures. No new
strict case passes. See the [pass-14 report](pass-14-summary.md) for the current
verification and remaining reconstruction gaps.

''' + needle)
for report in reports.values():
    for result in report['results']:
        if 'plus' not in result:
            continue
        pol = max(result[p]['complex_relative_l2'] for p in ('plus', 'cross'))
        projection = max(p['complex_relative_l2'] for p in result['projections'].values())
        row = f"| {result['case_id']} | {pol:.8g} | {projection:.8g} | {result['status']} |"
        text = re.sub(r'^\| ' + re.escape(result['case_id']) + r' \|.*$', row, text, flags=re.M)
start = text.index('## Reproduction and provenance')
intro, repro = text[:start], text[start:]
repro = repro.replace('verification/pass-12/', 'verification/pass-14/')
repro = repro.replace('rerun in pass 12', 'rerun in pass 14').replace('[pass-12 driver]', '[pass-14 driver]').replace('[pass-12 verification]', '[pass-14 verification]')
path.write_text(intro + repro)
print(f'Updated reports: {passed} passed, {skipped} skipped; {len(changed)} incremental code/fixture files.')
