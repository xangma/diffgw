# IMRPhenomXHM calibration critical path

Date: 2026-09-14. This is a clean-room inventory for the installed default
XHM release `122022`. It records the data needed to finish a production
implementation; it does not add a runtime route, fit coefficients, or a parity
threshold. No body of `xhm_fits.py`, LAL source, private audit cache, or
source-only calibration table was inspected for this inventory.

## Decision

The public binary reports the installed default release as `122022`, with the
default inspiral-amplitude provider controls observed as version `122018`,
variant `3`. The public paper and its ancillary mathematical assets are
version `arXiv:2001.10914v2` and describe the earlier calibration family. The
earlier `122018`/`122019` values are useful controls, but are not interchangeable
with `122022`: the audited contract says that the current release changes the
fit data, cuts, intermediate collocation behavior, and normalization.

No admissible public artifact containing the complete `122022` calibration
surfaces was found in the reviewed evidence. This is a record of “not found in
the checked sources,” not a proof that the artifact cannot exist. The missing
data are the blocker for exact current-release XHM parity. The equations,
normalization checks, node solver, and explicit provenance guard can be
implemented now with caller-supplied `122022` values.

The phase conclusion is narrower than the amplitude conclusion: the installed
`122022` change is an amplitude-calibration change, while the paper-era phase
model remains the retained four-condition construction for `(3,2)` (arXiv
`2001.10914v2`, Eqs. 46–50). The public paper supplies the phase ansatz, four
collocation locations, and mixing/alignment equations as a labeled old-phase
control. It does not supply release-specific numerical phase target fits or
redistribution authorization, so this retained algebra is not a claim of
`122022` data clearance.

## Version and evidence ledger

| Item | Version or identity | Evidence and hash | What it establishes | Limitation |
| --- | --- | --- | --- | --- |
| Installed public XHM default | release `122022`; inspiral-amplitude controls `122018`, variant `3` | Public binary setter observations recorded in [`xhm-inspiral-public.json`](verification/xhm-inspiral/xhm-inspiral-public.json), SHA-256 `2ea0c7d296dc9b3138cccbdce0433770bf940f71e197a1d031ce0928449329e1` | The release/version split that a production route must preserve | A setter result exposes behavior, not the calibration arrays or their license |
| Primary XHM equations | García-Quirós et al., `arXiv:2001.10914v2` | Public [paper](https://arxiv.org/abs/2001.10914v2); paper TeX hash recorded by the audit: `5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e` | Appendix-E PN polynomials, published ansatz, mode definitions, and old collocation conventions | Printed paper precision and the paper release do not identify the installed `122022` data |
| Public ancillary mathematical assets | paper v2 ancillary archive | The earlier clean-room report records public asset hashes, including `alpha222` `eb7708484832f2ca3a086b51bcc59f4fd2514c2e52633f26aa676c26a45ab781`, `alpha322` `1162a9205e597481938f29c0a76c857e3b17c0fec02de2a46471c95d3df3ab24`, and `alpha333` `c16b67ba0b66b5f6f0cf48d035d0a28fac2af211dc554fe1348d13cc3c964334`; see [`pass-2-higher-modes.md`](../2026-09-12-clean-room/pass-2-higher-modes.md), SHA-256 `be40838eeb3ea0a593214e7b56b12f331d13d6eee7c2ebe782705c67198b3467` | Public paper-era mathematical metadata exists | These assets are not identified as release `122022` data. Their bodies were not used here; public reachability does not settle redistribution or license status |
| Clean PN and collocation algebra | MIT clean helpers | [`xhm_amplitude_clean.py`](../../../src/torchwave/approximants/xhm_amplitude_clean.py), SHA-256 `d10bf39f21d58a31b5bdfd56242d9c73ba82e0db61e662ed73e4290cef0db5e4`; [`xhm_inspiral_assembly_release122022_clean.py`](../../../src/torchwave/approximants/xhm_inspiral_assembly_release122022_clean.py), SHA-256 `3a8ec52088ab24d0f618b5e2a6a6f4e5d04b10717c2f01706fad597234ff4bd1` | An independently implemented re-expanded PN baseline, `.5/.75/1` solver, geometric normalization, `q<20` guard, and explicit release/provenance contract | These helpers deliberately contain no fit values and therefore cannot supply the missing surfaces |
| Public XHM outputs | Paper-era/public API fixtures | Inspiral NPZ `/tmp/torchwave-xhm-inspiral-assembly-raw.npz`, SHA-256 `855036b69fdc006106d56a4cdcbac67609bdd703ff1faeddabc1375af95d25a8`; SPA NPZ `/tmp/torchwave-xhm-spa-public-raw.npz`, SHA-256 `c9200b6193957cfd45ec725a73fc16c7f369731e65a9ba782f75d8e78fde48e1`; amplitude NPZ `/tmp/torchwave-xhm-amplitude-public.npz`, SHA-256 `a7c879d4a0f8cc413a97fa23131930774c014f03d284af5ebcb2430a9700e4c5` | Reproducible black-box validation fixtures | Outputs are validation evidence only. Recovering calibration values from them would be oracle fitting |

The MIT calibration audit is limited to NRTidal and packaging metadata. Its
statement that the inherited PhenomD source-table gap is outside that audit is
also a boundary here; it is not a certification of XHM calibration data or of
the full source tree. See [`mit-calibration-audit.md`](mit-calibration-audit.md),
SHA-256 `722547757ab074845c1e738f248750b5ac4ebc60790332206237133646243d28`.

## Exact production input inventory

The names below are the fields used by the current inherited assembly where
that assembly has an explicit field. A `122022` provider may expose an
equivalent array, but a production adapter must record the mapping and units
instead of assuming that an older field has the same meaning.

### Common release and upstream inputs

* Release selection must remain explicit: `122022`. The observed public
  `122018` inspiral provider control is a compatibility diagnostic, not a
  substitute for the current release.
* The bounded current-release inspiral contract is `q < 20` and uses the
  mode-scaled boundary
  `f_ins(l,m) = (m/2) f_MECO22`. The clean adapter accepts `f_MECO22`, three
  geometric Fourier-strain amplitudes, and an explicit retained-node tuple.
* The complete waveform also needs the upstream 22-mode carrier, remnant mass
  and spin, `f_MECO22`, `f_ISCO22`, and Kerr QNM frequencies/damping times. These
  are carrier/QNM inputs rather than higher-mode calibration surfaces; their
  exact release-provider precision still has to be pinned when a production
  route is assembled.
* The inherited current assembly evaluates QNM functions for
  `21,22,32,33,44`, then divides by the fitted remnant mass. It uses `XAS` as
  the 22 carrier and applies the final Fourier factor
  `exp(-2 pi i f tc + i m phic)` after all mode joins (the code constructs
  `exp[-i(2 pi f tc - m phic)]`). Those conventions must remain common to
  every region.

### Inspiral amplitude, four requested higher modes

For each row, the absent item is the **release-122022 numerical calibration
array**, not the analytic PN baseline.

| Mode | Required release-122022 input | Current clean/inherited convention | Public admissible status |
| --- | --- | --- | --- |
| `(2,1)` | `A_geo_21[0.5,0.75,1.0]` at `0.5*f_MECO22`, with a region label and retained-node tuple | Re-expanded Appendix-E baseline in the current-release clean adapter; the paper-era diagnostic also has a non-re-expanded SPA branch valid to `q<40`, but that branch must not be reused for `122022` | Appendix-E equations are public and cleanly implemented; the three current geometric node surfaces and their region policy are absent |
| `(3,3)` | `A_geo_33[0.5,0.75,1.0]` at `1.5*f_MECO22`, with region label and retained-node tuple | Re-expanded Appendix-E printed baseline; odd-`m` node values are amplitudes and are folded to non-negative values by the clean solver | Equation and solver available; current-release node values and region masks are absent. The old provider disagreed by about `0.3%–0.5%` on one audited control, so this is not a decimal-rounding issue |
| `(3,2)` | `A_geo_32[0.5,0.75,1.0]` at `1.0*f_MECO22`, with region label and retained-node tuple | Re-expanded Appendix-E baseline with the named 32 compatibility sign variant; old 32 cut logic also depends on the 32/22 ringdown-frequency ratio, which is not part of the new clean release adapter | Equation and solver available; current-release node values, cut policy, and dynamic ringdown veto data are absent |
| `(4,4)` | `A_geo_44[0.5,0.75,1.0]` at `2.0*f_MECO22`, with region label and retained-node tuple | Re-expanded Appendix-E baseline with the named 44 compatibility sign variant | Equation and solver available; current-release node values and region masks are absent |

The clean `122022` solver uses pseudo-PN powers `(7/3, 8/3, 3)` and solves
the residual at the ascending fractions `.5, .75, 1`. It requires a provenance
record with `release=122022`, quantity
`geometric_fourier_strain_amplitude`, and those exact node fractions. It has
no automatic vetoes. This is implementable scaffolding, not a claim that the
caller-supplied values exist.

### Intermediate amplitude

The inherited current region assembly requires two fitted intermediate
targets per higher mode:

```
amp_int1_lm(eta,s1,s2,S), amp_int2_lm(eta,s1,s2,S)
```

The inherited names represent intermediate amplitude targets normalized by
the common 22 leading amplitude (`A^Inter_lm/A0`); the region solve converts
them to the inverse-`H` polynomial. A release provider may supply an equivalent
geometric target only with the conversion documented. The target frequencies are

```
f_int1 = f_ins_amp + (f_rd_amp - f_ins_amp)/3
f_int2 = f_ins_amp + 2*(f_rd_amp - f_ins_amp)/3.
```

For reference, the inherited `f_ins_amp` expression is

```
f_ins_amp(l,m) = (m/2) * (f_MECO22*eps_amp + |f_ISCO22-f_MECO22|*del_amp)
eps_amp(21,33,44) = 1,       eps_amp(32) = f_ring32/f_ring22
del_amp(21) = .75-.235*chi_eff-(5/6)*chi_eff**2
del_amp(33) = .75-.235*chi_eff-(5/6)*chi_eff
del_amp(32) = (.75-.235*|chi_eff|)*(f_ring32/f_ring22)
del_amp(44) = .75-.235*chi_eff.
```

The inherited mode-specific ringdown amplitude boundaries are

```
f_rd_amp(21) = .75*f_ring21
f_rd_amp(33) = .95*f_ring33
f_rd_amp(32) =       f_ring22
f_rd_amp(44) = .90*f_ring44.
```

Its phase boundaries are separately `f_ins_phi(l,m)=(m/2)*f_MECO22` and
`f_rd_phi=(f_ring21-f_damp21, f_ring22-.5*f_damp22,
f_ring33-f_damp33, f_ring44-f_damp44)` for 21, 32, 33, and 44 respectively.

This amplitude boundary is the paper-era MECO/ISCO interpolation; it is not
the clean `122022` inspiral boundary contract above. A current-release
implementation must obtain the release definition before reusing this
intermediate region. For every one of `21, 33, 32, 44`, the two numerical
target surfaces, their normalization, region masks, and any removed-node order
are absent. The inherited assembly's parameter-space, pole, and
ringdown-amplitude veto fallbacks therefore cannot be promoted as `122022`
policy.

### Intermediate phase

For `(2,1)`, `(3,3)`, and `(4,4)`, the current inherited field names are

```
phase_p1_lm, phase_p2_lm, phase_p3_lm, phase_p4_lm, phase_p6_lm.
```

They are solved as derivative targets at the five inherited points
`[C1,C2,E2,E3,E5]`. With `f_start = beta_eta*f_ins_phi` and
`f_end = f_ring_lm`, the six available points are

```
C1 = f_start
C2 = ((sqrt(3)+2)*f_start - (sqrt(3)-2)*f_end)/4
E2 = (f_end + 3*f_start)/4
E3 = (f_end + f_start)/2
E4 = (3*f_end + f_start)/4
E5 = (7*f_end + f_start)/8.
```

The public ancillary archive recorded by the prior report exposes author
assets named `p1` through `p5`, while the inherited assembly calls its fifth
target `p6`. The target-to-node and derivative/phase conversion, including the
paper Figure 8 versus Table VII choice, is unresolved. Thus all five target
surfaces for each of 21/33/44, their precise target convention, and the
`122022` region/version are absent. The public equations alone do not define
that conversion.

For `(3,2)`, the current inherited assembly requires

```
phase_p1_32, phase_p2_32, phase_p3_32, phase_p4_32
```

at the first four points `[C1,C2,E2,E3]`, then adds a ringdown join value and
derivative. The current inherited solve is a six-condition basis solve, but
the separate spheroidal construction below uses only three calibration
derivative targets. The four current-release intermediate targets, their
normalization, and the exact node/connection convention are absent.

The clean four-condition helper mirrors the retained paper-era spheroidal
ringdown system, but it is not wired into production: the current assembly
still supplies only three spheroidal derivative targets, fixes `alpha2_32` to
the 22 value, and then solves a different six-unknown intermediate system.
There is no independently sourced fourth target or rule authorizing that
reduction. The missing condition remains an explicit integration blocker;
interpolating or reusing the three targets would fabricate calibration data.

### Ringdown amplitude

For every higher mode, the current inherited assembly names two fitted
ringdown-amplitude quantities:

```
amp_alambda_lm(eta,s1,s2,S), amp_lambda_lm(eta,s1,s2,S).
```

It also needs the width factor `sigma_amp`: `amp_sigma_21(eta,s1,s2,S)` for
21 and fixed inherited constants `1.30` for 33 and `1.33` for 32/44. Whether
those constants and the 21 surface belong unchanged to release `122022` is
not documented by an admissible current-release artifact. The fit surfaces,
precision, sign/absolute-value convention, and the release-specific dynamic
ringdown-amplitude veto are absent for all four modes.

The mode QNM inputs needed to evaluate these functions are

```
qnm_fring_21, qnm_fdamp_21,
qnm_fring_22, qnm_fdamp_22,
qnm_fring_32, qnm_fdamp_32,
qnm_fring_33, qnm_fdamp_33,
qnm_fring_44, qnm_fdamp_44.
```

The Kerr QNM relation is a public analytic upstream ingredient, but the exact
provider interpolation/data precision selected by release `122022` is not
identified in the reviewed XHM calibration artifacts. It must be pinned as a
separate upstream contract; public waveform outputs cannot be used to infer it.

### Ringdown phase and 32 mixing

The non-mixing 21/33/44 branches require the shared 22 ringdown fits

```
phase_rd_alpha_lambda(eta,s1,s2,S), phase_rd_alpha2(eta,s1,s2,S),
```

then scale `alpha2_lm` by `fdamp22/fdamp_lm`. The shared surfaces and their
release-specific precision are absent. The inherited phase branch also keeps
an absolute-mode phase offset and a 21 sign decision that the clean-room
report has deliberately left uncertified.

The 32 spherical/spheroidal branch additionally names

```
phase_32_s_deltadphi(eta,s1,s2,S)
phase_32_s_deltaphi(eta,s1,s2,S)
phase_32_s_dphi_p1(eta,s1,s2,S)
phase_32_s_dphi_fring32(eta,s1,s2,S)
phase_32_s_dphi_p4(eta,s1,s2,S)
```

and the complex mixing entries represented by the inherited names
`mixing_alpha222`, `mixing_alpha223`, `mixing_alpha322`, and
`mixing_alpha333`. The earlier public assets provide only three paper-era
real-part mathematical files and do not certify the complete complex
index/sign convention. The inherited `alpha2_32 = alpha2_22` choice is an
explicit three-condition compatibility choice; the paper-era Table IX
description instead indicates four data/unknowns. A release-complete route
needs the fourth condition or a citable rule selecting the reduced system,
plus all four current mixing functions and alignment frequencies. Those
release/versioned data and conventions are absent.

## Region and version gaps

The following older rules are available in the clean paper-era helper and are
useful only as labeled controls:

* 21: for `q<8`, remove geometric node amplitudes below `0.2`; remove the
  middle node for a non-monotonic three-node sequence.
* 33: for `1<q<1.2`, `chi1<-0.1`, `chi2>0`, remove the final node.
* 32: for `q>2.5`, `chi1<-0.6`, `chi2>0`, or both spins below `-0.9`, remove
  the two highest nodes; a separate dynamic ringdown-amplitude veto is
  inherited.
* 44: no printed node veto was identified in the cited section.

These rules establish the old paper-era control behavior. They do not fill the
current `122022` region table. In particular, the installed inherited source
also contains parameter-space and pole fallback decisions whose calibrated
inputs are not independently certified. The current release adapter therefore
requires an explicit retained tuple and rejects omitted policy rather than
silently selecting one.

Precision is a separate blocker from availability. The audit found that the
old 33 provider nodes differ from public paper-era reconstruction by roughly
`0.3%–0.5%` on one control, demonstrating changed calibration rather than
printed-decimal rounding. A binary output or a rounded paper coefficient is
not a substitute for full-precision `122022` surfaces.

## Implement now versus external dependency

Implementable without new data:

1. Keep release and provenance explicit (`122022`, geometric Fourier strain,
   `.5/.75/1` nodes, `q<20`).
2. Keep the independent Appendix-E PN baselines, named 32/44 compatibility
   variants, mode-scaled `f_ins`, geometric normalization, and linear
   pseudo-PN solver.
3. Add a provider interface whose required records are the arrays and region
   labels listed above, with shape/unit/version validation and no automatic
   fallback to the old provider.
4. Validate supplied records with synthetic collocation recovery and public
   outputs only as post hoc diagnostics; do not derive records from those
   outputs.

Required before a production `122022` route can be completed:

1. A versioned, redistributable or otherwise explicitly authorized calibration
   artifact containing the four-mode inspiral geometric node arrays and the
   exact retained-node/region table.
2. The four-mode intermediate amplitude targets and the complete current
   normalization/connection rules.
3. The 21/33/44 phase target arrays and target-to-derivative convention; the
   retained paper-era 32 four-point spheroidal target values and any
   release-specific replacement or version declaration.
4. Per-mode ringdown amplitude surfaces, width conventions, dynamic veto
   values, and all shared 22 phase surfaces.
5. The complete 32 spheroidal alignment/mixing artifact, including the fourth
   condition or an explicit published reduced-system rule, with complex
   signs/indexes and full precision.
6. A release-pinned statement of QNM/remnant/upstream interpolation precision
   where the XHM provider differs from the clean carrier contracts.

## Precise unsent external request

If an author or release maintainer must be contacted, request only the
following, without asking for runtime source:

> Please provide the calibration-data release used by IMRPhenomXHM default
> release `122022` (including the observed inspiral controls `122018`, variant
> `3`), with a citation and redistribution/license terms. For each of
> `(2,1)`, `(3,2)`, `(3,3)`, and `(4,4)`, please provide full-precision
> inspiral geometric Fourier-strain values at `.5`, `.75`, and `1.0` times the
> release `f_ins`, the retained-node and parameter-region table, intermediate
> amplitude targets, phase targets and their exact node/derivative conversion,
> ringdown amplitude/phase targets, and the complete `(3,2)` spheroidal
> alignment and complex mixing data. Please state the release-specific
> `f_ins`, `f_int`, and `f_rd` definitions, normalization, signs, units,
> precision, missing-node/pole fallback rules, dynamic vetoes, and whether
> every value is authorized for redistribution under the project license.

No request was sent. Public availability and redistribution permission are
separate questions; neither is inferred from a public URL or from the MIT
status of the surrounding clean code.
