# SEOBNRv4 native-generator diagnostics: numerical gate failed

This development-only evidence extends the [first campaign](../2026-09-22/README.md).
The recorded source snapshot was unchanged during both runs. The numerical
campaign uses scalar CPU `float64`, a 30+20 solar-mass nonspinning binary, and
starting orbital frequency `M Omega=0.04`. Exact inputs, source hashes, solver
costs, installed LAL binary identities, and per-check gates are in
[qualification.json](qualification.json). Its `git_head` is the parent commit
at run time; the source hash records the edited generator under test.

| Refined comparison | Coarse to medium complex L2 | Medium to fine complex L2 | Medium to fine amplitude L2 | `1e-6` gate |
| --- | ---: | ---: | ---: | --- |
| ODE tolerances `rtol=1e-11,1e-12,1e-13` | 1.902e-5 | 8.748e-6 | 3.331e-7 | **Failed** |
| Output grid `dt/M=0.5,0.25,0.125` | 1.648e-7 | 6.082e-6 | 2.585e-7 | **Failed** |

The paired absolute tolerances were `1e-12,1e-13,1e-14`. All five bounded
solves completed. Peak/match time changes at the tightest ODE pair were
`1.526e-5 M`, within their separate `0.05 M` gate. The output-grid error rose
at the last refinement, so these runs show sensitivity rather than demonstrated
convergence. Changing `dt` also changes the current adaptive initial step and
coarse/fine restart sampling. Event-peak location uses an explicit `1e-5 M`
bisection tolerance. Those are investigation targets, not established causes
of the measured errors. No threshold or waveform coefficient was adjusted to
make the gate pass.

The [whole-waveform diagnostic](whole_waveform.json) used the tightest
`dt/M=0.125`, `rtol=1e-13`, `atol=1e-14` settings and the pinned public
LALSuite `lal=7.7.1`, `lalsimulation=6.2.1` `SEOBNRv4` binary. The
[comparison inputs](whole_waveform_inputs.json) add one aligned-spin row
(`chi1=+0.2`, `chi2=-0.1`) to the nonspinning campaign row. Each complex `(2,2)`
mode is placed at its own measured amplitude peak (`t=0`, peak phase zero);
no shift, amplitude scale, or coefficients are optimized. Metrics use common
support; the JSON retains raw full-reference-support metrics too.

| 30+20 solar masses | Common peak-relative support | Complex relative L2 | Amplitude relative L2 | Phase RMS | Real overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chi1=chi2=0` | `[-310,150] M` | 0.02171 | 0.002219 | 0.02420 rad | 0.999764 |
| `chi1=+0.2`, `chi2=-0.1` | `[-330,150] M` | 0.01337 | 0.001665 | 0.01298 rad | 0.999911 |

Raw full-support complex errors were 1.987 and 1.864, respectively, before
the stated time/phase anchoring. The aligned diagnostics isolate shape
differences under that convention; they are not an optimized match, a
detector-weighted mismatch, a physical-fidelity test, or a parity claim.
Numerical convergence was **not** prechecked for either row. The second row
has no dedicated convergence campaign, so it is exploratory. These results
do not qualify the generator or ROM training.

Reproduce from the repository root with the recorded runtime and binary
versions (the first command is expected to exit 1 because its gate fails):

```sh
PYTHONPATH=src python -m dev.native_rom.qualify_generator \
  --atol-levels 1e-12 1e-13 1e-14 --rtol-levels 1e-11 1e-12 1e-13 \
  --compare-lal --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out /tmp/seobnrv4-qualification.json
PYTHONPATH=src python -m dev.native_rom.compare_whole_waveforms \
  --qualification /tmp/seobnrv4-qualification.json \
  --inputs dev/native_rom/evidence/2026-09-23/whole_waveform_inputs.json \
  --json-out /tmp/seobnrv4-whole-waveform.json
```

The next numerical experiment should isolate event-root precision, adaptive
initial step, and restart sampling, then rerun the declared gates. Independent
review of the full spin Hamiltonian and all 35 factorized mode amplitudes is
still required before these diagnostics can support a native SEOBNRv4 claim.
