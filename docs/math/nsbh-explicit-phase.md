# NSBH explicit-gauge phase composition, v1

## Scope and status

This is a conditional component gate for the aligned-spin, nonspinning-NS
frequency-domain PhenomNSBH phase. It is not a whole-waveform qualification.
The default legacy, frequency-grid-dependent affine gauge, remnant fit, QNM
fit, and disruption-dependent amplitude are outside this gate.

## Inputs, conventions, and options

The BH is body 1 and the NS is body 2: `m1 >= m2 > 0` in solar masses,
`chi2 = 0`, `Lambda1 = 0`, `Lambda2 >= 0`. Frequency and positive reference
frequency are in Hz; time is in seconds. The complex carrier is
`exp(-i psi)` with a native NSBH overall minus sign. Only explicit
`coalescence` and `reference_frequency` phasing conventions are covered.
The latter anchors the PhenomD baseline at `f_ref`; the tidal term is not
independently re-anchored. This is deliberately distinct from the legacy
default. The QNM pair `(M_total f_RD, M_total f_damp)` and PhenomD fit rows
are supplied numerical conditioning inputs, not results of this component.

## Mathematics and components

Thompson et al., [arXiv:2002.08383v2, Sec. II.2](https://arxiv.org/html/2002.08383v2),
specify a PhenomD BBH phase baseline augmented by the updated NRTidal
phase. Thus `psi_NSBH(f) = psi_D(f; f_RD, f_damp) + psi_T(f; Lambda1=0,
Lambda2)`. `psi_D` uses the C1 joins of Khan et al.,
[arXiv:1508.07253v2](https://arxiv.org/html/1508.07253v2), with its
declared QNM pair. `psi_T` is Dietrich et al.,
[arXiv:1905.06011v2, Eqs. (17)-(21)](https://arxiv.org/pdf/1905.06011v2),
with the mass-weighted tidal combination of Wade et al.,
[arXiv:1402.5156v1, Eq. (5)](https://arxiv.org/pdf/1402.5156v1).
The explicit phase gauge and `-pi/4` coalescence offset follow
[imrphenomd-intrinsic.md](imrphenomd-intrinsic.md). The NSBH plus carrier is
`-A_+(f) exp(-i psi_NSBH(f))`; this check divides out `|h_+|` and requires
nonzero output, so it does not assess the amplitude magnitude.

| Component ID/version | Source | Status | Scalar reference | Native check |
| --- | --- | --- | --- | --- |
| `nsbh.phase.explicit-composition.v1` | Thompson Sec. II.2; Khan phase; Dietrich Eqs. (17)-(21) | Conditional | `diffgw_reference.nsbh_phase.phase_expr` | `test_nsbh_explicit_phase_math_reference.py` |

## Implementation mapping

The scalar expression composes independent `imrphenomd_waveform` and
`nrtidalv2_phase` references. The Torch and JAX NSBH implementations add
their PhenomD baseline and NRTidalv2 phase before constructing the complex
polarizations. Native final-spin and QNM computation is used only to supply
the conditioning QNM pair to the scalar reference. Both sides use the same
declared, higher-precision PhenomD coefficient rows already tested by the
PhenomD component gate; those extra digits are not independently certified
by the printed paper table.

## Evidence and limitations

Frozen before comparison: cases `(m1,m2,chi1,chi2,Lambda2)` are
`(7,1.4,0,0,600)` and `(12.6,1.4,0.4,0,0)`; dimensionless frequencies
`Mf = (0.010, 0.018, 0.035, 0.070)` and `Mf_ref = 0.020`; `tc=0.003 s`,
`phic=0.23 rad`, and both explicit gauges. CPU FP64 Torch and JAX unit
phasors must satisfy absolute complex error `<= 3e-7` against the 70-digit
scalar expression, with no numerical phase/time alignment or binary fit.
The scalar expression must agree between 70 and 90 digits within `1e-30`
in phase radians. This samples smooth phase regions; it does not prove
boundary differentiability, remnant/QNM provenance, amplitude fidelity,
legacy-gauge behavior, or agreement with NR data. A passing gate remains
conditional on its supplied inputs.
The [validation receipt](../validation/nsbh_explicit_phase_20260927.md)
records the `len` CPU FP64 results.

## Reproduce the checks

Run `pytest -q tests/test_nsbh_explicit_phase_math_reference.py` with
SymPy, mpmath, Torch, JAX, and CPU FP64 support. Changes to the phase
composition, gauge, conditioning inputs, or tolerance require a new version
and updated evidence receipt before comparison. The PhenomD and NRTidalv2
component checks remain independent required neighbors.
