# Clean TEOB TD-to-FD orchestration

This directory records the bounded, opt-in clean assembly implemented in
`teob_conditioned_waveform_clean.py`.  The supported sector is scalar,
zero-spin, zero-tide TEOBResumS with mode `(2,2)`:

`native support and sampling → PA → adaptive/fixed continuation → clean TEOB
state carrier → explicit NQC targets → UIB2016 remnant → ringdown assembly →
clean TD conditioning and Fourier transform`.

Both attribution choices are required at the call site.  The physical smoke
uses `radial_acceleration_policy=frozen-mobility` and
`horizon_policy=leading-order`; the adapter also exposes the named
`conservative`, `zero`, and `current` choices.  No choice is inferred, and
the canonical waveform driver is unchanged.  The 2017 NQC target provider is
the clean published provider with its provenance label returned in the result.

Run the default 30+20 solar-mass smoke with:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
scripts/diagnose_teob_conditioned_waveform_clean.py \
--radial-acceleration-policy frozen-mobility \
--horizon-policy leading-order
```

The command writes diagnostic.json, reusable native-output.npz,
catalog-output.npz, public-fd-output.npz, and a stage log.  The report
records host, command, PID, support and Fourier
sampling, native inverse-Kepler radius, PA and continuation events, NQC
matrices/conditions/residuals, ringdown attachment in total-mass and
final-mass units, stage arrays, and source SHA-256 values before and after the
run.  The output epoch is explicitly the coordinate start of the assembled
ringdown series.  `physical_epoch_seconds` is separately derived from the
first accepted continuation orbital-frequency decrease event, matching the
public epoch convention.

The public binary fixture
/tmp/torchwave-public-teob-q1p5-r20-dt0p5M.npz is read only for provenance;
the report checks SHA-256
`09eaa8b00288c94947b2bb4b40ada7780dc51ac98491a48cbb8f60ed50ab686c` and
records its metadata.  Its support-conditioning and generic/direct
projection evidence are linked by hash in the report.  The smoke does not
align, fit, search parameters, truncate support, or claim full trajectory
validation.  The public fixture starts from `19.999999999999996`, while this
adapter derives its own start radius from native support as
max(r_min, omega_start**(-2/3)); that distinction is retained in the
report.

The projection seam is checked at inclinations `0` and `0.7` and phases `0`
and `0.4`; the clean direct and generic conventions cancel to the published
floating-point tolerance.  The lower-level NQC phase-lift regression also
covers sparse carrier phase jumps larger than `pi` before unwrapping.

The cache-backed
`scripts/diagnose_teob_public_fd_clean.py` diagnostic evaluates the public
`SimInspiralFD(..., approximant=TEOBResumS)` generic TD-to-FD path for the
native 30+20 request and the 1.4+1.4, 30-Hz catalog request.  It first
verifies the frozen native cache hashes, then compares exact shared 0.5-Hz
bins without alignment or optimization.  It reports full bands 20--1024 Hz
and 30--1024 Hz, respectively, together with catalog subbands 20--256 Hz
and 30--500 Hz.  The resulting arrays and metrics are in
`public-fd-sim-inspiral-output.npz` and
`public-fd-sim-inspiral-report.json`.

For provenance, the report also records the earlier
`SimInspiralChooseFDWaveform` attempt as a superseded native-FD dispatcher.
That call returns `Invalid argument` because TEOBResumS is exposed as
TD-only by the installed LALSimulation build; `SimInspiralFD` is the
supported public path used for the actual comparison.
