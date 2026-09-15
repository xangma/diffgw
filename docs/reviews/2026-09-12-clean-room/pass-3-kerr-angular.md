# Angular Teukolsky solution and ringdown integration

`kerr_angular.py` reconstructs the spin −2 spectral eigenproblem in
[Cook–Zalutskiy, arXiv:1410.7698v2](https://arxiv.org/pdf/1410.7698v2),
Eqs. (4), (49)–(56). The freshly retrieved PDF has SHA256
`33973382059461835989057a47e93336e5233cccede48529b4b19cdc306d7330`.
No numerical implementation source or binary residuals enter this solver.

Multiplication by cos(theta) is represented by its spherical-harmonic
recurrence. Squaring that matrix includes an extra basis function before
truncation. The angular matrix is `A0 − c² x² − 4 c x`, with
`c = a M omega`. Continuation from `c=0` selects the requested eigenbranch;
the final eigenvector has unit norm and a real-positive coefficient at the
requested spherical degree. The implementation requires finite `|c|<=2`
and a sufficiently small spectral tail. Its default maximum degree is 18.
Gradients pass through the final eigensolution, not its discrete branch label.

Seventeen focused tests cover the spherical limit, explicit harmonics,
angular differential-equation residuals, sphere normalization, first-order
perturbation theory, complex gradients, batches, resolution and domain checks.
Six untouched author samples independently check separation constants and
mixing coefficients. Those samples come from Gregory Cook's
[v4 numerical database](https://doi.org/10.5281/zenodo.16801267), under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).
The fixture records original files, SHA256 hashes, dataset keys, row indices
and phase rotation. It contains attributed information from Cook's database;
it is a modified subset, normalized to the paper's real-positive convention.

An independent clean-room agent checked matrix signs, indexing, recurrence,
phase normalization and branch continuation. An eight-step continuation
agreed with 128 steps on 257 complex boundary rays for fifteen low modes.
This is numerical evidence within the tested domain, not a general guarantee
at eigenvalue crossings or for arbitrary higher modes.

## Public ringdown API

`evaluate_qnm`, both `KerrRingdown` domains and `get_ringdown_waveform` now use
`qnm_backend="author"` by default, with the attributed numerical representation
described in [the QNM report](pass-3-qnm.md). The approximate inherited tables
remain available with `qnm_backend="berti"`. Author spins outside
`[0, 0.99999]` are rejected instead of extrapolated. Remnant mass must be
finite and positive. Physical units follow `omega=Re(M omega)/M_sec` and
`tau=M_sec/[-Im(M omega)]`.

The existing spherical angular basis remains the default.
`angular_basis="spheroidal"` selects the spectral solution and its equatorial
partner. The mirror uses `-c*`, as derived independently in
[the projection note](pass-3-ringdown-projection.md). Both time and frequency
domains preserve the existing mode-amplitude normalization; no energy-to-mode
amplitude convention is inferred from an oracle. Parameter batches include
spin, orientation, mode amplitudes and phases, with sampling on the last axis.

The combined QNM, angular, ringdown and Fourier tests pass: **72 tests**.
They include independent numerical time integration of the real waveform in
both bases, Hermitian symmetry, physical mass scaling, parameter gradients,
and multidimensional batches versus scalar calls.

## Independent binary diagnostics

The parent verifier subsequently called public LALSimulation 6.2.0 binary
APIs; the implementers did not use these outputs. Reproduce with:

```sh
PYTHONPATH=src python scripts/verify_kerr_components.py --json-out kerr-components.json
```

[Recorded results](verification/pass-3/kerr-components.json) compare 60
complex frequencies with `SimIMREOBGenerateQNMFreqV2fromFinal` at five spins.
Maximum relative complex difference is `3.405e-6`. This particular API rejects
the `(3,2)` family, which is recorded as unsupported, not a passing check.

For the `(2,2)` angular function, the verifier supplies the same public Leaver
frequency to both solvers and converts LAL's polar-angle normalization to
sphere normalization by dividing by `sqrt(2*pi)`. Separation constants agree
within `9.4e-9`. After removing only an arbitrary constant eigenfunction phase
for diagnosis, angular relative differences are approximately `1.10e-4`,
`8.37e-4`, and `1.24e-3` at spins 0.3, 0.7 and 0.95. The phase is recorded
only in the report; it is not applied in native code or acceptance tests.

These component comparisons do **not** establish full LAL ringdown waveform
parity. Native angular solutions satisfy the independent differential
equation and author data, but measurable binary differences remain. The
catalog retains an unresolved full-waveform reference, including its
amplitude and phase conventions.
