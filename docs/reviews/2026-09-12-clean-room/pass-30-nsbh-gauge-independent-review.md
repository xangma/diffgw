# Pass 30: IMRPhenomNSBH gauge review

## Scope

This review checks the IMRPhenomNSBH affine phase and time gauge without
reading or translating LALSuite or PyCBC implementation source.  The evidence
used here is Thompson et al. (2020), the Dietrich NRTidal papers, the PhenomD
paper, TorchWave's independent implementation, and public waveform API calls
used only as black boxes.

## Result

The papers specify the intrinsic carrier and tidal phase, but they do not
uniquely specify the global affine phase/time origin used by a waveform API.
In particular, they do not resolve the inherited NSBH upper-frequency time
anchor or the omitted-reference default.  The existing TorchWave legacy
default therefore remains unchanged: an omitted `f_ref` uses its documented
30 Hz compatibility reference, and the affine residual found in the previous
audit is not evidence for fitting a new slope or intercept.

One exact public API defect was independently resolved.  The public
`IMRPhenomNSBH` call accepts `f_ref=0` and uses the lower edge of the generated
positive-frequency band as the reference sentinel.  Black-box calls with
`flow=20 Hz` give identical waveforms for `f_ref=0` and `f_ref=20 Hz`; with
`flow=30 Hz`, `f_ref=0` equals `f_ref=30 Hz`; changing to 40 Hz produces only
the expected constant phase rotation.  Before this pass, TorchWave sent zero
into logarithmic inspiral phase expressions, producing NaNs.  This was an
exact runtime defect rather than a fitted correction.

The implementation now resolves an explicit nonpositive reference to the
lowest positive frequency in the supplied grid before evaluating the carrier
phase.  The omitted-reference path still resolves to 30 Hz.  The same
resolved reference is passed to the reference-frequency and legacy branches;
the coalescence branch remains independent of `f_ref`.

## Phase composition

The independent algebra is

```
phase = phase_phenomd + phase_tidal
```

PhenomD owns the reference-frequency subtraction and the caller's affine
`tc`/`phic` terms.  The NRTidal composition adds its intrinsic tidal phase;
the NSBH tidal Padé term is not separately re-anchored.  Subtracting the
tidal value at `f_ref` would introduce a new convention not fixed by
Thompson et al. or Dietrich et al. and would not explain the carrier affine
residual.  No slope/intercept was fitted to oracle residuals.

## Verification

Focused clean-room tests passed after the fix:

```
29 passed in 1.51s
```

The new regression test verifies finite output and exact equality between
`f_ref=0` and the lowest positive grid frequency, and preserves the documented
omitted-reference/30 Hz behavior.  `git diff --check` also passes.

