# NSBH tidal derivative diagnostic

`nsbh_tidal_derivative_diagnostic.py` compares the black-box phase ratio
`h(lambda2)/h(lambda2=0)` with the direct NRTidalv2 Eq. 18 construction.  The
mass-weighted deformabilities are `10.8147694514` (catalog) and
`25.1028806584` (holdout), using physical masses in seconds.  Both signs of
the Fourier phase construction are recorded; the negative sign is the
equation-consistent one.

At the frozen references and `f_final=1024`, the negative-sign analytic
construction has complex L2 `0.0050597` native / `0.0050582` binary (catalog)
and `0.1027990` native / `0.1035962` binary (holdout).  Nonaffine phase RMS is
`8.23e-10` / `2.48e-9` rad for catalog and `7.76e-9` / `9.27e-6` rad for
holdout.  The residual is therefore not an unexplained affine carrier alone;
the direct unanchored tidal term leaves a material holdout mismatch at high
frequency.

The published Eq. 18 phase and derivative contain no `f_final` argument, so
their predicted ratio between endpoint calls is exactly unity (derivative
difference zero).  On common `30<=f<512` samples, the observed 512-versus-
1024 tidal-ratio complex L2 is `4.57e-5` native and machine precision binary
for catalog, and `6.13e-4` native / `4.49e-3` binary for holdout.  The phase
differences are affine to about `1e-13` rad nonaffine RMS.  This identifies an
endpoint-dependent affine convention in the assembled waveform ratio; Eq. 18
does not prescribe that convention, so no correction or anchor is inferred.

The JSON records all endpoints `512,1024,2048`, references (catalog `30,27,60`;
holdout `40,27,60`), both backends, both Fourier signs, derivatives, metrics,
versions, and source hashes.

The black-box `_legacy_phase_offsets` callable was available with signature
`(m1, m2, s1, s2, mf_ref, mf_anchor, remnant_mass, remnant_spin)` and was
temporarily wrapped and restored by descriptor assignment.  Calls for every
frozen case, endpoint, reference, and lambda state are recorded, including
`mf_ref`, `mf_anchor`, remnant inputs, and outputs.  For example, catalog
`512→1024` has `mf_anchor=0.0161398487372→0.0322796974745`, corresponding to
`512→1024 Hz` after division by the captured total mass in seconds.  The
NRTidalv2 derivative difference evaluated at the endpoint, last sample, and
captured native anchor is `+3.7514536106e-4 rad/Hz` for this pair; all three
candidate anchor choices are explicitly present in JSON.  The observed tidal
ratio slope is `+1.6400909e-7 rad/Hz` native (binary is near machine precision),
so the fixed analytic candidate
`exp(+i (f-f_ref) [psi'_T(anchor512)-psi'_T(anchor1024)])` does not explain the
observed ratio.  No fitted shift or runtime change was made.

The diagnostic also runs two process-local black-box substitutions for native
NSBH: freeze only `remnant_mass` to the lambda-zero call, then freeze both
`remnant_mass` and `remnant_spin`.  For catalog at `f_final=1024,f_ref=30`,
the negative-sign unit-phase L2 falls from `1.5202e-4` (original) to
`1.26e-13` under either substitution; the corresponding phase slope falls
from `-2.6495e-7` to `9.0e-18` rad/Hz.  Thus the affine tidal-ratio phase is
fully explained by the remnant-mass change in the native phase-offset
callable; spin freezing adds no measurable change for this case.  Equivalent
records are present for holdout, every endpoint, and every frozen reference.
Raw complex, amplitude, and unit-phase metrics are retained separately.
For holdout at `f_final=1024,f_ref=40`, unit-phase L2 is `3.1227e-3` original,
`1.4151e-2` with mass-only freezing, and `6.07e-14` with mass-plus-spin
freezing.  Its phase slope correspondingly reaches `2.21e-17` rad/Hz only
when both remnant quantities are frozen.
