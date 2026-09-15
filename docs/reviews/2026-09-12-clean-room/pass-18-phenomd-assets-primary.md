# Pass 18: PhenomD coefficient and provenance audit

This is a bounded clean-room audit of IMRPhenomD source semantics. The primary
sources were read before comparing the native implementation. No parity output,
residual, binary, translated implementation, notebook, generated implementation,
git history, or verification directory was used.

## Primary sources and provenance

The primary papers and source archives are:

- Khan et al., *Frequency-domain gravitational-wave model of nonprecessing
  black-hole binaries*, arXiv:1508.07253v2: [record](https://arxiv.org/abs/1508.07253v2),
  [source archive](https://arxiv.org/src/1508.07253v2), DOI
  [10.1103/PhysRevD.93.044007](https://doi.org/10.1103/PhysRevD.93.044007).
  The downloaded source archive is `khan1508.tar`, SHA-256
  `e70cde9d5e9ee92e4592c4c93152f6eb2410264358486f6a68064fb03b39b1cf`; its
  extracted `PhenomPaper2.tex` is SHA-256
  `ed0d9ca1031def94a8df3d2964967dd88d50aa179bd12fd2bbf1cfb067070b44`.
- Husa et al., *Frequency-domain gravitational waves from nonprecessing black-hole
  binaries. I. New numerical waveforms and the anatomy of the signal*,
  arXiv:1508.07250v2: [record](https://arxiv.org/abs/1508.07250v2),
  [source archive](https://arxiv.org/src/1508.07250v2), DOI
  [10.1103/PhysRevD.93.044006](https://doi.org/10.1103/PhysRevD.93.044006).
  The downloaded source archive is `husa1508.tar`, SHA-256
  `79dd5b277fd99d8ff3576800fb274b296bea8ae1553f134a015205f40fae5b4a`; its
  extracted `PhenomPaper1.tex` is SHA-256
  `8a8a4a9820cba1edd935dd500a6c194ac728ae1f5bd716b8f262295d1bcbfc8d`.

The Khan archive contains 61 files with extensions `.tex`, `.bbl`, and `.pdf`;
the Husa archive contains 24 files with `.tex`, `.bbl`, `.pdf`, and `.png`.
Neither archive contains `.m`, `.mat`, `.dat`, `.csv`, `.json`, `.nb`, or other
coefficient/QNM data files. Thus the permitted primary archives do not provide
a full-precision PhenomD coefficient supplement or a QNM interpolation table.

## Coefficient map: supported facts

Khan Eq. (31), `PhenomPaper2.tex` lines 1494--1507, defines the map in the
unscaled `chi_PN` variable, expanded around `chi_PN=1`:

```
Lambda = lambda_00 + lambda_10 eta
       + (chi_PN-1)(lambda_01 + lambda_11 eta + lambda_21 eta^2)
       + (chi_PN-1)^2(lambda_02 + lambda_12 eta + lambda_22 eta^2)
       + (chi_PN-1)^3(lambda_03 + lambda_13 eta + lambda_23 eta^2).
```

The eleven-column ordering is printed at `PhenomPaper2.tex` line 2623:
`lambda_00, lambda_10, lambda_01, lambda_11, lambda_21, lambda_02,
lambda_12, lambda_22, lambda_03, lambda_13, lambda_23`. The published Table V
rows are rounded decimal values (lines 2624--2642); for example the beta2 row
has exactly eleven entries, ending `..., -3.41293, 25.5724, -54.408`
(line 2636).

The native `src/torchwave/approximants/phenomd_unrounded_fits.py` has nineteen
named vectors (`rho1..3`, `v2`, `gamma1..3`, `sigma1..4`, `beta1..3`, and
`alpha1..5`), each with eleven entries, and
`src/torchwave/approximants/imrphenomd.py:71-87` evaluates the same ordering and
powers as Eq. (31). The native vectors' extra digits are numerically plausible
continuations of the printed rows, but no permitted Khan/Husa source asset
establishes their provenance or exact rounding-to-binary relationship.

There is a source-paper accounting ambiguity. Khan says at lines 1467--1477
that four amplitude and four phase coefficients are constrained analytically,
leaving 17 mapped quantities, while Table V prints 19 rows, including beta1
and alpha1. Lines 1553--1561 say the time/phase-shift constants are fixed by
continuity. The native runtime maps beta2/beta3 and alpha1..alpha5, computes
beta0/beta1 and alpha0/alpha1 by C1 matching, and retains a raw alpha1 for its
phase gauge (`imrphenomd.py:369-388`, `536-589`, `652-660`). This is a concrete
implementation choice, but the permitted paper text does not resolve why the
Table V beta1/alpha1 rows coexist with the stated 17 free mapped quantities.
No source-complete coefficient correction is justified.

## Remnant and QNM data: supported facts and blocker

Husa states at `PhenomPaper1.tex:821-827` that dimensionless `M omega_RD` is
obtained by interpolating the Berti data set and that the binary frequency is
scaled by `M_f = M-E_rad`. It gives no interpolation knots, interpolation order,
fit coefficients, endpoint rule, or machine-readable data in the arXiv source.
The final-spin fit is Eq. (3.6), lines 877--889, and the radiated-energy fit is
Eqs. (3.7)--(3.8), lines 903--915. The native constants in
`phenomd_unrounded_fits.py:310-350` preserve the same polynomial/rational
semantics with extra digits, but those extra digits likewise have no permitted
primary provenance.

The native `imrphenomd.py:143-151` calls its own `evaluate_kerr_qnm_220` and
scales the returned remnant frequency by `M/M_f`. This has the same dimensional
scaling stated by Husa, but equivalence to the paper's unspecified interpolation
cannot be proven from the allowed assets. The missing Berti table/interpolation
metadata is the primary blocker to an exact source-complete QNM correction.

The native clamps the fitted final spin to `[-0.999,0.999]` before the QNM call
(`imrphenomd.py:127-128`). Husa instead says the fit respects the Kerr limit
(lines 888--889) and gives no operational clamp. This is another runtime safety
policy, not a source-defined calibration rule; it should remain unchanged until
an admissible implementation specification is found.

## Phase joins, phase anchor, and cutoff

Khan's phase semantics are explicit. The merger-ringdown derivative and
integral are Eqs. (20)--(21), `PhenomPaper2.tex:912-950`; the fit window is
`[0.45,1.15] f_RD` and the fitted merger-ringdown phase is used above
`0.5 f_RD` (`:928-953`). The intermediate ansatz is Eq. (22),
`eta phi'_Int = beta1 + beta2/f + beta3/f^4`, over `[0.018,0.5 f_RD]`, with
fit window `[0.017,0.75 f_RD]` (`:994-1018`) and integrated form at
`:1022-1032`. The final piecewise phase and transitions are Eq. (34),
`:1563-1575`: `f1=0.018`, `f2=0.5 f_RD`, with C1 continuity determining the
constant and linear terms in the intermediate and merger-ringdown regions.
The native joins at exactly these boundaries (`imrphenomd.py:399-400` and
`:551-589`), so the phase-region semantics are source-supported.

The paper does not choose a unique absolute phase/time anchor. It explicitly
says at `PhenomPaper2.tex:1553-1561` that the full phase is defined up to an
arbitrary time and phase shift; those shifts are absorbed in the inspiral
constant/linear terms, while the other region constants are fixed by C1
continuity. The native `slope_peak` subtraction (`imrphenomd.py:652-660` and
`:888-903`) and reference-frequency subtraction are therefore gauge/convention
choices. They are not specified by the primary paper, but are mathematically
within its stated time/phase freedom. A source-exact anchor rule cannot be
recovered from the permitted assets.

There is likewise no paper-defined hard `Mf=0.2` model cutoff. The full model
formula is the piecewise phase/amplitude construction above. In the time-domain
appendix, Khan instead introduces an external Planck taper and says one typically
chooses its lower support relative to the desired `f2` (`PhenomPaper2.tex:2388-
2403`); it also observes that the amplitude has dropped by several orders of
magnitude above `Mf>0.25` (`:2405-2409`). The native default `F_CUT=0.2` and
zeroing path (`phenomd_unrounded_fits.py:33-34`, `imrphenomd.py:718-744`) are
explicitly documented in native comments as an inherited operational boundary,
not a fit equation. They cannot be replaced by a source-exact number from the
allowed primary material.

## Result and implementation decision

No exact source-complete mismatch was found. The native Table V map shape/order,
phase transition frequencies, C1-join equations, and remnant scaling agree with
what the primary papers actually specify. The unresolved items are precisely the
ones the papers omit: provenance of the unrounded coefficient digits, the
Berti-QNM interpolation data/algorithm, the reason Table V prints constrained
beta1/alpha1 rows, and an absolute phase/time anchor or hard frequency cutoff.
Accordingly, no runtime or test file was changed and no parity or residual test
was run. The supported follow-up is to obtain an admissible author/source asset
that supplies the missing full-precision vectors and QNM interpolation metadata;
until then, changing those native values or replacing the `Mf=0.2` boundary
would be speculation rather than a clean-room correction.
