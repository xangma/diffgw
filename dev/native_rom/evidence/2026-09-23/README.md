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
bisection tolerance. No threshold or waveform coefficient was adjusted to
make the gate pass.

The subsequent [grid-coupling diagnostic](grid_coupling.json) held these two
numerical choices fixed separately. It used the same 30+20 nonspinning row
with `atol=1e-14`, `rtol=1e-13`, and exact nested waveform samples. The
default `dt/M=0.125` solve restarted at `164.875 M`, whereas the `0.5 M`
baseline restarted at `165 M`. Relative to that baseline:

| `dt/M=0.125` choice | Complex relative L2 |
| --- | ---: |
| Default first step and restart | 5.916e-6 |
| First step fixed at `0.5 M` only | 6.365e-6 |
| Restart fixed at `165 M` only | 9.817e-8 |
| Both fixed | 0 |

A `0.25 M` run with both fixed was also identical on nested samples. The
recorded source hash was unchanged during all six solves. Thus the restart
sample shift accounts for most of the measured grid discrepancy in this row;
the diagnostic overrides do **not** change the default published-convention
route or rescue its failed qualification gate. In particular, the ODE
tolerance error in the first campaign was `8.748e-6`, and no fixed-restart
run establishes correctness of the equations or full waveform. Reproduce
this diagnostic with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_grid_coupling --json-out
/tmp/seobnrv4-grid-coupling.json`.

An additional [tighter qualification run](tight_qualification.json) used
`rtol=(1e-13,1e-14,1e-15)` and paired
`atol=(1e-14,1e-15,1e-16)`, retaining the same default restart rule and
declared `1e-6` complex-error gate. All five solves completed and the recorded
source hash stayed unchanged. The ODE complex differences were `2.159e-6`
then `1.158e-5`; the grid differences were `1.789e-7` then `2.613e-5`.
Both checks failed, including their required decreasing-error condition. The
peak times differed by at most `1.526e-4 M` across the refined grid pair,
still within the separate `0.05 M` event-time limit. The nonmonotonic results
do not establish an asymptotic error regime or identify a precision floor.
This run did not change the waveform or promotion status. Reproduce it with
the `qualify_generator` command below, replacing both tolerance triples with
the tighter values and writing a separate JSON output.

The [regional ODE diagnostic](ode_regions.json) repeated its three ODE levels
at `dt/M=0.5`, partitioning each unaligned complex waveform into fixed time
windows. In the finest pair, the complex relative L2 was `1.158e-5`, versus
an amplitude relative L2 of `3.657e-7`; phase RMS was `1.488e-5 rad`.
Approximately 1% of squared complex error was before the `165 M` restart,
21% from `165–300 M`, 21% from `300–310 M`, 33% from `310–320 M`, and 24%
thereafter. The earlier tolerance pair had the same broad distribution, though
smaller errors. Thus the discrepancy is phase dominated and already present
in the inspiral, not confined to attachment or ringdown. These are descriptive
windows, not qualification gates or a determination of the numerical cause.
The source hash remained fixed. Reproduce with
`PYTHONPATH=src:. python -m dev.native_rom.diagnose_ode_tolerance --json-out
/tmp/seobnrv4-ode-regions.json`.

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

The next numerical experiment should compare the orbital phase trajectory at
fixed times with an independent integration method, then rerun the declared
gates without tuning the thresholds to these observations.
Independent review of the full spin Hamiltonian and all 35 factorized mode
amplitudes is still required before these diagnostics can support a native
SEOBNRv4 claim.
