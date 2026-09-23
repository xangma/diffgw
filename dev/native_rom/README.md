# Native ROM development

This branch-only workspace contains experimental EOB assemblers, the retired
pilot surrogate, and development checks. It is excluded from release packages.
The prototypes are not registered runtime models, qualified SEOBNRv4 generators,
or replacements for the official ROM data. No trained asset is promoted here.

The [first frozen qualification run](evidence/2026-09-22/README.md) failed the ODE
convergence budget. The generator is not qualified for independent ROM training.
An opt-in [fixed-restart numerical diagnostic](evidence/2026-09-23/README.md)
passes both convergence gates for one nonspinning row; the default route still
fails. Its sampled-peak public-binary comparison has a `0.02171` complex
relative L2 difference; [sub-sample peak anchoring](evidence/2026-09-23/README.md)
reduces it to about `0.0068–0.0069`. The same diagnostic choices fail both
numerical gates for an aligned-spin row; its sampled-peak difference of
`0.01336` falls to about `0.0078–0.0081` with sub-sample anchoring. These
comparisons do not validate the physical generator.

The later [fixed-trajectory output-sampling check](evidence/2026-09-23/README.md)
finds identical accepted trajectories across three grids and much smaller
Hermite-versus-DOP853 differences than natural-cubic differences through
`165 M`. In a predeclared
[six-row exploratory parameter campaign](evidence/2026-09-23/README.md),
both start-frequency rows pass the numerical gates, but four mass-ratio/spin
rows fail at least one. In particular, the strong aligned-spin row's refined
ODE complex error is `6.57e-6`, above the `1e-6` budget. The provisional
generator remains unqualified; no ROM training or release promotion follows.

The versioned [SEOBNRv4 paper reference](reference/SPEC.md) pins published
calibration, Hamiltonian, selected flux, leading initial-state, peak, and
ringdown components and maps applicable checks to the frontends. Its
[manifest](reference/manifest.json) marks unresolved generator components
explicitly. Resolve those entries and validate the complete generator before
extending the ROM work or promoting any model.

## Numerical evidence

From the repository root, using a development environment with Torch and NumPy:

```sh
PYTHONPATH=src python -m dev.native_rom.qualify_generator \
  --json-out /tmp/native-generator-evidence.json
```

The default scalar row is `mass1=30`, `mass2=20` solar masses, `chi1=chi2=0`,
and dimensionless orbital `omega_start=0.04`. Supply `--inputs /path/rows.json`
for a JSON list of rows with exactly those five keys. Run `--print-plan` with
the same command to inspect all settings without generating a waveform.

Each row runs five Torch CPU float64 solves with one thread: three ODE tolerance
pairs `(atol, rtol) = (1e-9, 1e-8), (1e-10, 1e-9), (1e-11, 1e-10)` at
`dt=0.5 M`, then the tight pair at `dt=0.25 M` and `0.125 M`. Ringdown is
`150 M`. Scientific choices stay fixed at `generic-leading-ell8`,
`bohe_total_energy`, no Pan-v8 or positive-spin-540 option, and final-spin fit
orders `(3, 4)`; no comparison selects coefficients or options.

The final pair in each refinement sequence must have direct complex and
amplitude relative L2 errors at most `1e-6`, nonincreasing complex error,
and peak/match/boundary time differences at most `0.05 M`. These are numerical
source budgets, not a physical-accuracy certification. Comparisons use exact
nested samples, include unmatched tails by zero filling, and never fit time,
phase, or amplitude. Changing the grid also changes the prototype's initial
step and restart sampling, so this checks the complete generator's sensitivity.

Each solve has a 600-second wall-clock cap, 200,000 derivative-evaluation cap,
20,000 accepted-step cap and 10,000 rejected-step cap. The default campaign has
at most five such solves per row. `--max-run-seconds` and `--max-evaluations`
can reduce resource bounds; a bounded-out run fails rather than passing with
missing evidence. Runtime is machine and input dependent. Ctrl-C records a
failed campaign. Use `/tmp` for evidence and logs.

An optional unaligned public-binary diagnostic requires exact module versions;
the local development environment currently supports:

```sh
PYTHONPATH=src python -m dev.native_rom.qualify_generator \
  --compare-lal --lal-version 7.7.1 --lalsimulation-version 6.2.1 \
  --json-out /tmp/native-generator-lal-evidence.json
```

This calls `SimInspiralChooseTDWaveform(..., SEOBNRv4)` only through its public
API, records wrapper and shared-library hashes, and does not inspect LAL source.
The native orbital-frequency peak defines its time origin; the LAL epoch stays
unchanged. Identical numeric phase input does not establish identical convention
semantics. This reference comparison reports discrepancies and makes no parity
pass claim. Requested reference errors fail the campaign.

JSON records exact configuration, input and source hashes, versions, run costs,
per-check errors and gates, and explicit false physics/training/production
qualification flags. Nonfinite outputs, exceptions, changed sources and failed
gates produce a nonzero exit. The runner exports no training arrays or ROM
asset. Its adaptive prototype currently uses independently computed QNM values
with black-box-inferred spline knots; resolving that provenance is still needed
for a wholly literature-specified generator.

## Remaining promotion stages

The independent ROM construction follows the published reduced-basis and
parameter-interpolation methods of [Pürrer (2014)](https://doi.org/10.1088/0264-9381/31/19/195010)
and [Pürrer (2016)](https://doi.org/10.1103/PhysRevD.93.064041).
[Bohé et al. (2017), Section VII](https://arxiv.org/html/1611.03703#S7)
describes the SEOBNRv4 ROM construction and its parameter patches. Independently
trained tables would define our own approximation to a qualified source model;
they would not become the authors' official ROM tables.

1. Qualify the physical generator against the published equations and independent
   references over the intended domain, resolve convention/provenance gaps, and
   establish gradients and numerical convergence. This runner covers only scalar
   CPU forward numerical stability; convergence alone cannot validate physics.
2. Freeze and hash that generator, construct independent ROM training data, and
   qualify an independent ROM against held-out generator waveforms, including
   parameter boundaries and interpolation derivatives. Identify it as an
   independently trained approximation rather than the official ROM asset.
3. Establish the same waveform, dtype/device, batching and derivative contracts
   in Torch and JAX, complete release validation and provenance review, and only
   then consider production registration and packaging.

Focused runner tests: `PYTHONPATH=src python -m pytest -q
dev/native_rom/tests/test_qualification.py`.
