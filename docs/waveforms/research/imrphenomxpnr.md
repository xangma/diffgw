# IMRPhenomXPNR: component evidence and qualification

## Verdict

The DiffGW `IMRPhenomXPNR` name currently selects a **provisional native
composition**, not a qualified implementation of the complete published
PhenomXPNR model.  This page qualifies a narrower result: the printed
single-spin map, fit polynomial, guarded opening/precession-angle transforms,
merger-ringdown ansatzes and higher-mode frequency map have an independent
stdlib scalar reference and agree with both Torch and JAX.  During this audit,
the implemented `B3` transform was corrected from `B2 * Lambda_B3` to the
published `chi sin(theta) [exp(Lambda_B3) - 200]` relation.

Full-waveform qualification remains failed.  The public route uses
reference-frequency rather than transition-frequency spins in the effective
single-spin map, does not implement the published angle attachment and analytic
continuations, and combines its angles with the native symmetric XHM carrier
rather than the paper's tuned XHM-CP carrier, effective ringdown prescription,
and dominant-mode asymmetry.  The inherited numeric calibration tensors also
lack an independently established source and redistribution record.

The sources pinned for this ledger are:

- [Hamilton et al., *PhenomXPNR*, arXiv:2507.02604v1](https://arxiv.org/html/2507.02604v1), 3 July 2025, CC BY 4.0;
- [Thompson et al., *IMRPhenomXO4a*, arXiv:2312.10025v1](https://arxiv.org/html/2312.10025v1), 15 December 2023, CC BY 4.0;
- the public [BAM precessing-waveform catalogue](https://data.cardiffgravity.org/bam-catalogue/).

The earlier Hamilton et al. 2021 “final twist” paper is useful model history,
but it is not the specification of the 2025 XPNR composition audited here.

## Published component scope

The following identifiers separate equation fidelity from complete-model
composition.

| Component | Published prescription | DiffGW evidence | Status |
| --- | --- | --- | --- |
| `XPNR-MAP` | XPNR Eqs. (3)–(6), evaluated with spins evolved to the SpinTaylor stopping/transition frequency | Scalar/Torch/JAX algebra agrees when supplied the same spins | Component pass; public caller supplies the wrong epoch |
| `XPNR-FIT` | XO4a Eq. (35), `Lambda = sum lambda_pqr eta^p chi^q cos(theta)^r`, plus XPNR Eq. (22) freezing | Scalar/Torch/JAX values agree; tensor duplication is not independent provenance | Equation pass; assets unresolved |
| `XPNR-ANGLES-MR` | XO4a Eqs. (25)–(34), XPNR Eqs. (15)–(16), including stated guards | Scalar values and frequency derivatives agree in Torch/JAX | Component pass after `B3` correction |
| `XPNR-ATTACH` | XPNR Eq. (19) for alpha and Eqs. (20)–(21) for beta, matching inspiral/merger-ringdown values and derivatives | Current alpha is a value-only Hermite-style blend; current beta sequence is not the published construction | Fail |
| `XPNR-CONT` | XPNR Eqs. (17)–(18) analytic exterior continuations and Eq. (22) fit-coordinate freeze | Current public route window-blends the angle tracks | Fail |
| `XPNR-CARRIER` | XPNR Eqs. (7)–(8): tuned XHM-CP deviations and effective ringdown frequencies | Independent scalar equations exist; neither is wired into the waveform | Fail |
| `XPNR-ASYM22` | XPNR Eqs. (10)–(13), unequal `m=+2` and `m=-2` coprecessing modes | Native XHM carrier generates negative-m content by symmetry | Fail |
| `XPNR-HMMAP` | XO4a Eqs. (50)–(52), used by XPNR Eqs. (23)–(24) | Scalar/Torch/JAX piecewise values and joins agree | Component pass; not a complete carrier pass |
| `XPNR-TWIST` | SpinTaylor inspiral, attached NR-tuned angles, minimal rotation, and multimode rotation | Native XPHM composition is executable and differentiable | Engineering evidence only |

## Exact map, fit, and angle relations

For `m1 >= m2`, XPNR Eqs. (3)–(6) map the two transition-frequency spins to

```text
chi_eff       = (m1 chi1z + m2 chi2z) / (m1 + m2)
chi_parallel  = (m1 + m2) chi_eff / m1
chi_perp      = |m1^2 chi1_perp + m2^2 chi2_perp| / m1^2
chi           = sqrt(chi_parallel^2 + chi_perp^2)
cos(theta_LS) = chi_parallel / chi .
```

The zero-spin angle is undefined by the equation; the scalar test oracle uses
`cos(theta_LS)=0` only as a documented numerical extension.  The fit-coordinate
continuation freezes `eta` at `0.09` and `chi` at
`0.80 - 0.20 exp(-((q-6)/1.5)^8)` when those boundaries are crossed.

Writing `s = chi sin(theta_LS)`, the printed XO4a transforms are

```text
A1 =  s Lambda_A1^2                 B0 = theta_f - s Lambda_B0
A2 = -s Lambda_A2^2                 B1 =  s exp(Lambda_B1)
A3 =     Lambda_A3^2                B2 = -s exp(Lambda_B2)
A4 =     Lambda_A4^2                B3 =  s [exp(Lambda_B3) - 200]
                                       B4 =  Lambda_B4^2
                                       B5 = -Lambda_B5^2 .
```

The implementation and scalar oracle apply the paper's guards: `A1 >= 0`,
`A2 <= 0`, `A3 >= 1e-5`, `A2 >= -pi^2 sqrt(A3)`, and `B4 >= 175`.  The
merger-ringdown ansatzes are

```text
alpha(f) = -[A1/f + A2 sqrt(A3)/(A3 + (f-A4)^2)] + alpha_bar
beta(f)  = B0 + (B1 + B2 f + B3 f^2)/(1 + B4 (f+B5)^2) .
```

The XO4a text bounds beta with an arctangent construction.  The native route
instead clips beta to `[0, pi]`; therefore bounded output is not evidence for
the paper's bounding prescription.  Likewise, XO4a Eq. (38) prints
`cos((x-x_b)/x_w)` without the factor of pi required to reach zero at the outer
endpoint.  Both native backends use the endpoint-completing shifted cosine,
equivalent to `cos(pi (x-x_b)/(2 x_w))^2`, for `q: 8.5 -> 12` and
`chi: 0.85 -> 1.2`.  Tests label this as backend agreement on an explicit
interpretation, not exact transcription of the printed line.

## Carrier and reduction relationships

The full 2025 model modifies the coprecessing carrier as well as its Euler
angles.  XPNR Eq. (7) deforms each calibrated XHM-CP coefficient by

```text
lambda'_k = lambda_k + chi sin(theta_LS) u_k,
```

while Eq. (8) uses

```text
omega'_lm = omega_lm
            - m [1 - |cos(beta_RD)|] (omega_22 - omega_21).
```

The `u_k` values are not printed in the paper; Appendix A of XO4a locates them
in LALSuite source.  XPNR Eqs. (10)–(13) additionally introduce dominant
`+2/-2` coprecessing-mode asymmetry.  Consequently, “PNR angles + symmetric
XHM” is not an exact carrier reduction of the published full model.

The precise component reductions are:

- At zero transverse effective spin, `s=0`: `A1=A2=0`, the `B0` correction and
  `B1`, `B2`, `B3` vanish, and Eq. (7) gives `lambda'_k=lambda_k`.
- If the aligned ringdown also has `|cos(beta_RD)|=1`, Eq. (8) gives
  `omega'_lm=omega_lm`.  The published dominant-mode asymmetry vanishes in the
  corresponding nonprecessing limit.  With consistent angle gauges, the full
  published construction then reduces to its aligned XHM carrier.
- The current DiffGW public route takes an early nonprecessing branch and
  numerically collapses to its **native** XHM waveform.  That verifies the
  native composition only; it does not validate the missing XHM-CP terms.
- Outside the angle-fit region, the published merger-ringdown angles revert
  through Eqs. (17)–(18), while some carrier deviations fade separately from
  `q=10` to `q=20` and the effective-ringdown change remains.  The test window
  and the optional `match >= 0.99` threshold therefore define neither a
  physical calibration boundary nor an observed accuracy domain.

For a higher mode, the XO4a map is `f_22=2f/m` below the PN join, a line joining
the endpoint values in the intermediate interval, and
`f_22=f-(f_RD,lm-f_RD,22)` above the merger-ringdown join.  This frequency map
does not supply the absent carrier deformation or asymmetry.

## Calibration assets and terms

| Asset | Published availability and terms | Repository treatment |
| --- | --- | --- |
| XPNR/XO4a equation text | arXiv HTML is CC BY 4.0 | Equations independently encoded in `diffgw_reference.xpnr_angles`; attribution retained |
| BAM 80-waveform single-spin calibration set | Public catalogue: `q={1,2,4,8}`, `chi={0.2,0.4,0.6,0.8}`, tilts `30,60,90,120,150` degrees; [minimal Zenodo record 7673796](https://doi.org/10.5281/zenodo.7673796) and [complete record 7677297](https://doi.org/10.5281/zenodo.7677297), CC BY 4.0 | Not bundled and not re-fit in this change; the discrete training design is not an accuracy bound |
| Numeric angle-fit and carrier-deviation tables | Not printed in either paper. XO4a Appendix A points to LALSuite commit `08b90494b3a4a7dc966ca108e6c98eaa3f6e18a7`; LALSuite is GPL-2.0-or-later | No LAL source was inspected, copied, translated, or fitted. Pinned binaries are comparison oracles only |
| Existing DiffGW `A1..A4`, `B0..B5`, `Bf` tensors | Independent origin and redistribution terms are unresolved | Retained but not promoted. Torch/JAX equality proves duplication only; `Bf` is currently not wired into the public route |

The canonical little-endian float64 hash of the existing tensors, with names
and shapes included, is
`c7794b653f781bbba45b97170fdac9fb6c89de6cc9c404e4942908ce1f820d68`.
The Torch and JAX source-file hashes are respectively
`9d5d77749d187f64447f1ed0501311370b5c97a3e8a471de71ced315ce0394fd`
and
`5801aa497b7ba10d389d8da5ca21dc27dc072c953a71706157211ca5fe2a6ea3`.
Different source hashes are expected because the modules use different tensor
constructors; exact numeric-array equality is checked separately.

## Evidence matrix

| Evidence class | Result | What it establishes |
| --- | --- | --- |
| Paper fidelity | Partial pass | Map, polynomial, guarded transforms, MR ansatzes, carrier-deformation equation, effective ringdown equation, and HM map have scalar encodings; attachment, continuation, beta bound, carrier/asymmetry composition remain open |
| Scalar-reference agreement | Pass for scoped components | 46 local cases cover fit values, transforms, ansatz values/derivatives, map, window interpretation, HM joins and reductions |
| Torch/JAX parity | Pass for scoped components | Both backends agree with scalar values and derivatives; a dense JAX grid is recorded in the qualification JSON |
| External parity | Not qualified | The four optional LAL/PyCBC cases are black-box acceptance gates only; their `0.99` threshold is not an observed lower bound or domain claim |
| Physical validation | Not reproduced | The papers report BAM/NRSur comparisons; this change does not compare DiffGW waveforms with those NR data |
| Full waveform | Fail | Missing published components plus inherited XHM's recorded strict catalog `0/1` and holdout `0/1` failures prevent promotion |

Machine-readable outcomes, versions, commands, and still-failed gates are in
[`imrphenomxpnr-qualification.json`](imrphenomxpnr-qualification.json).

## Promotion gates

Promotion requires all of the following without relaxing existing thresholds:

1. Establish a redistributable, independently traceable numeric calibration
   asset, or reproduce the fit from cited CC BY data with a documented method,
   uncertainties, and holdouts.
2. Evolve both spins to the stated transition and feed that state into Eqs.
   (3)–(6); test integration convergence and transition placement.
3. Implement the exact alpha/beta attachments, beta bound, exterior
   continuations, and minimal-rotation convention with value and derivative
   checks at every join.
4. Implement and separately test XHM-CP deformation coefficients, effective
   ringdown frequencies, dominant-mode asymmetry, and their nonprecessing and
   exterior reductions.
5. Pass same-option Torch/JAX complex-mode and both-polarization checks, then
   the existing strict full-waveform gates (`relative L2 <= 1e-6`,
   `mismatch <= 1e-8`) on declared catalog and holdout cases.
6. Reproduce physical validation against identified NR waveforms before making
   a physical-accuracy statement.  Optimized-polarization matches must remain a
   separate diagnostic.

Until then, the precise supported claim is “paper-evidenced XPNR angle
components with scalar/Torch/JAX agreement,” not “sovereign NR-calibrated
IMRPhenomXPNR.”
