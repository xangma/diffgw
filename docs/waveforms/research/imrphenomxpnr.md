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
The printed alpha/beta connection, exterior-continuation, final quadratic
beta-rescaling, two-spin beta taper, and optimal-emission-direction map now
also have isolated scalar/Torch/JAX kernels. They are not wired into the
public waveform because their input tracks, connection frequency, and terminal
state are not independently fixed by the available sources.

Full-waveform qualification remains failed.  The public route uses
reference-frequency rather than transition-frequency spins in the effective
single-spin map, does not implement the published angle attachment and analytic
continuations, and combines its angles with the native symmetric XHM carrier
rather than the paper's tuned XHM-CP carrier, effective ringdown prescription,
and dominant-mode asymmetry.  The inherited numeric calibration tensors also
lack an independently established source and redistribution record.

The sources pinned for this ledger are:

- [Hamilton et al., *PhenomXPNR*, arXiv:2507.02604v1](https://arxiv.org/html/2507.02604v1), 3 July 2025, CC BY 4.0;
- [Colleoni et al., *SpinTaylor precession*, arXiv:2412.16721v1](https://arxiv.org/abs/2412.16721v1), 22 December 2024, and [Phys. Rev. D 111, 104019](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.104019); the arXiv deposit has a non-exclusive distribution license and is used as specification evidence only;
- [Thompson et al., *IMRPhenomXO4a*, arXiv:2312.10025v1](https://arxiv.org/html/2312.10025v1), 15 December 2023, CC BY 4.0;
- [Hamilton et al., *The final twist*, arXiv:2107.08876v1](https://arxiv.org/abs/2107.08876v1), 19 July 2021, CC BY 4.0; and
- the public [BAM precessing-waveform catalogue](https://data.cardiffgravity.org/bam-catalogue/).

The 2021 paper supplies a partial generation procedure for the 40-waveform
predecessor fit.  It is not the specification or output of the later
80-waveform XO4a retune used by the 2025 XPNR composition.

## Published component scope

The following identifiers separate equation fidelity from complete-model
composition.

| Component | Published prescription | DiffGW evidence | Status |
| --- | --- | --- | --- |
| `XPNR-MAP` | XPNR Eqs. (3)–(6), evaluated with spins evolved to the SpinTaylor stopping/transition frequency and projected relative to instantaneous `Lhat` | Scalar terminal-state map is rotation-invariant; Torch/JAX component algebra agrees when supplied the same aligned-frame spins | Geometry pass; public caller supplies the wrong epoch and no terminal state |
| `XPNR-FIT` | XO4a Eq. (35), `Lambda = sum lambda_pqr eta^p chi^q cos(theta)^r`, plus XPNR Eq. (22) freezing | Scalar/Torch/JAX values agree; tensor duplication is not independent provenance | Equation pass; assets unresolved |
| `XPNR-ANGLES-MR` | XO4a Eqs. (25)–(34), XPNR Eqs. (15)–(16), including stated guards | Scalar values and frequency derivatives agree in Torch/JAX | Component pass after `B3` correction |
| `XPNR-ATTACH` | Final Twist Eqs. (41)–(45) for the beta taper/emission direction and XPNR Eq. (19) for alpha and Eqs. (20)–(21) for beta, matching inspiral/merger-ringdown values and derivatives | Standalone kernels reproduce the published forms, the XO4a denominator correction, and frontend derivatives; input-track construction and public connection remain absent | Component forms pass; full route fail |
| `XPNR-CONT` | XPNR Eqs. (17)–(18) analytic exterior continuations and Eq. (22) fit-coordinate freeze | Standalone Eq. (17)–(18) kernels reproduce synthetic published-form functions and derivatives; current public route still window-blends the angle tracks | Component pass; public route fail |
| `XPNR-CARRIER` | XPNR Eqs. (7)–(8): tuned XHM-CP deviations and effective ringdown frequencies | Independent scalar equations exist; neither is wired into the waveform | Fail |
| `XPNR-ASYM22` | XPNR Eqs. (10)–(13), unequal `m=+2` and `m=-2` coprecessing modes | Native XHM carrier generates negative-m content by symmetry | Fail |
| `XPNR-HMMAP` | XO4a Eqs. (50)–(52), used by XPNR Eqs. (23)–(24) | Scalar/Torch/JAX piecewise values and joins agree | Component pass; not a complete carrier pass |
| `XPNR-TWIST` | SpinTaylor inspiral, attached NR-tuned angles, minimal rotation, and multimode rotation | Native XPHM composition is executable and differentiable | Engineering evidence only |

## Exact map, fit, and angle relations

For `m1 >= m2`, XPNR Eqs. (3)–(6) map the two transition-frequency spins to

```text
chi_eff       = (m1 chi1_parallel + m2 chi2_parallel) / (m1 + m2)
chi_parallel  = (m1 + m2) chi_eff / m1
chi_i_parallel = chi_i . Lhat
S_i_perp      = m_i^2 (chi_i - chi_i_parallel Lhat)
chi_perp      = |S_1_perp + S_2_perp| / m1^2
chi           = sqrt(chi_parallel^2 + chi_perp^2)
cos(theta_LS) = chi_parallel / chi .
```

The zero-spin angle is undefined by the equation; the scalar test oracle uses
`cos(theta_LS)=0` only as a documented numerical extension.  The fit-coordinate
continuation freezes `eta` at `0.09` and `chi` at
`0.80 - 0.20 exp(-((q-6)/1.5)^8)` when those boundaries are crossed.

## Spin-at-attachment evidence boundary

The papers determine the terminal-state geometry, but not a complete
independently reproducible route from reference spins to that state.  XPNR
requires Eqs. (3)–(6) at the frequency where the merger-ringdown angles are
computed and says this frequency is related to the PN spin-precession stopping
frequency, for which it gives no closed form.  The SpinTaylor paper specifies
3.5PN TaylorT4 phasing, 3PN spin effects and fourth-order Runge--Kutta--Fehlberg
integration until a stepping condition is reached.  It does not publish the
complete coefficient system, RKF pair/controller/tolerances/interpolation,
exhaustive stop conditions, or an exact selection rule for the terminal sample.

XPNR also uses several nearby but distinct frequencies: `0.97 f_max^PN` and
`0.99 f_max^PN` in the alpha continuation, `0.97 f_max^PN` and
`0.98 f_max^PN` in the beta continuation, plus a separately constructed beta
attachment frequency.  None is unambiguously identified as the spin-sampling
epoch.  Equating one of them with that epoch would be an implementation choice,
not a paper transcription.

The scalar reference therefore accepts caller-supplied terminal dimensionless
spins `chi_i = S_i/m_i^2` and `Lhat` in any common Cartesian frame, performs
the instantaneous projection, and is tested for rotation invariance and
agreement with the z-aligned component formula.  These are algebraic component
checks, not physical validation of a terminal state.  The reference
deliberately does not evolve the state or choose the stop event.  The current
Torch/JAX route instead passes
reference-frequency spin components into the map; its SpinTaylor helper returns
only six angle/frame quantities.  Its `0.98 f_MECO` transition, frequency
clamps, and fixed 200-node RK4 evolution are recorded as project-specific
choices, not paper fidelity.  A source-level regression test keeps this boundary
visible until the caller is replaced deliberately.

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

## Isolated angle joins and continuations

The [scalar reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/xpnr_angles.py) and the
[Torch](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/pnr_attachment.py) and
[JAX](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/pnr_attachment.py) component kernels
take caller-supplied PN/MR boundary values and frequency derivatives. They do
not infer a SpinTaylor stopping event, fit coefficients, or a public dispatch.
The [equation test](https://github.com/xangma/diffgw/blob/main/tests/test_xpnr_attachment_math_reference.py)
compares their outputs against independently chosen coefficients in each of
the paper's printed forms and checks both frontend derivatives.

Before the final quadratic rescaling, Final Twist Eqs. (44)–(45) define
`iota_osc = iota_2spin - iota_single` and

```text
iota(f) = iota_single(f) + cos^2(pi f / (2 f_c)) iota_osc(f),  f <= f_c
        = iota_single(f),                                      f > f_c .
```

The isolated kernel is exactly two-spin at zero frequency, exactly
single-spin at `f_c`, and has zero taper derivative at that join. Final Twist
Eqs. (41)–(43) then map the opening angle to the optimal emission direction
using its printed `c_0,c_2,c_3,d_0,d_2,d_3` polynomials. XO4a explicitly
corrects the 2021 expression by inserting a factor of two in the denominator;
the scalar/Torch/JAX implementations and a frozen checkpoint exercise that
corrected form. These kernels accept the already-constructed angle tracks and
`f_c`; they do not infer those missing model inputs. The emission-direction
kernel takes caller-supplied effective spin components `chi_perpendicular` and
`chi_parallel`, as defined by the terminal single-spin map. This is the
published `chi sin(theta_LS)` and `chi cos(theta_LS)` dependence, while avoiding
an undefined angle derivative at zero spin.

For Eq. (19), multiply the printed `alpha = a0 f² + a1 f + a2 + a3/f`
by `f`; the resulting cubic is fixed by `f alpha` and
`d(f alpha)/df = alpha + f alpha'` at both joins. Eq. (20) is a cubic fixed
directly by beta values and derivatives at both joins. These are Hermite
interpolants of the published polynomial spaces, rather than new blends.

For Eq. (17), use `u = 1/f²`: alpha is quadratic in `u`, fixed by values at
`0.97 f_max^PN` and `0.99 f_max^PN` and the derivative at the latter.
For Eq. (18), subtract `beta_RD`, multiply by `exp(kappa f) f²` with
`kappa = 2 pi Delta f_damp`, and use `u = 1/f`: the result is quadratic in
`u`, fixed by values at `0.97 f_max^PN` and `0.98 f_max^PN` and the
derivative at the latter. The implementation centers the exponential at the
latter frequency to reduce overflow/underflow risk; this only rescales the
three coefficients and leaves the paper's function unchanged.

For Eq. (21), `k(f)=1+k1 f+k2 f²` multiplies the already-tapered,
emission-direction beta. Given that base beta and its derivative at the
merger-ringdown connection, two linear conditions fix `k1,k2` so the final
value and derivative match the target there. The constant `1` gives the
stated low-frequency unity limit. The scalar reference rejects a zero base
beta at the join, for which the coefficients are not uniquely determined by
this construction; the frontend kernels assume a qualified nonzero input.

The tests use synthetic coefficients rather than inherited calibration
tables. They establish form and differentiable alignment, not that the public
waveform chooses the correct boundary data, composes the preprocessing, or
reproduces a calibrated XPNR waveform.

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

## Published generation procedure and reproducibility boundary

The 2021 paper documents how the predecessor angle surfaces were constructed.
It fits each waveform's angle ansatz with SciPy `curve_fit` using the
Levenberg--Marquardt algorithm, then fits those per-waveform coefficients over
`(eta, chi, cos(theta_LS))` with `mvpolyfit`.  It gives the alpha fit interval
`[f_dip-0.0225, f_dip+0.0075]`, the beta interval `f_inf +/- 0.03`, the
sequential fit orders `A4,A1,A2,A3` and `B1,B2,B3,B5,B4`, and a 39-term ceiling
for the 40 calibration cases.  The cited `positive` v1.0 archive contains the
`mvpolyfit`/`gmvpfit` tooling under an MIT license (the Zenodo record labels its
license only as `other-open`).
That predecessor uses a `(4,2,5)` ambient polynomial box.  XO4a Eq. (35)
expands the spin order to the `(4,4,5)` shape of the current arrays, so the
historical procedure cannot by itself generate or identify the later tensors.

XO4a separately states that its carrier-deviation variables used a
“significantly refined” `gmvpfit` with `estatol=0.001`.  That statement occurs
in the carrier section, not the angle-retuning section, so it is not evidence
that the same fitter or setting generated `A1..A4`, `B0..B5`, or `Bf`.

XO4a expands the spin order of the polynomial, retunes the angles to all 80
BAM waveforms, and prints the transforms and basis, but does not deposit the
per-simulation fitted angle targets, derived QA Euler-angle series, fit
script/run configuration, selected sparse terms, or output tensors.  In
particular, the papers do not pin the sampling, weighting, initial values,
constraints, sequential-refit state, or software environment needed to
reproduce the final decimals.  Appendix A instead directs readers to final
values in a named LALSuite file and commit.  This audit did not inspect that
source.

The arXiv source packages were also inspected. Final Twist v1 has 51 files,
SpinTaylor v1 has 39, and XPNR v1 has 22; the respective archives contain
manuscripts/figures and, for XPNR, only a build manifest beyond the paper.
They contain no machine-readable terminal prescription, calibration targets,
fit configuration, or coefficient tensors. The XPNR archive SHA-256 is
`558b71b710e7e1e0eaa69efcdbc43e4a79de24ce7b02a5df7ed9173ac1cfba36`.

The published assets therefore support the equations, the raw NR inputs, a
historical method, and a fitting tool.  They do not independently reproduce or
license the inherited DiffGW tensor values.  A new fit from the BAM data would
be a separate calibration project with its own declared choices, uncertainty
analysis, and holdouts—not evidence for the provenance of the current arrays.

## Calibration assets and terms

| Asset | Published availability and terms | Repository treatment |
| --- | --- | --- |
| XPNR/XO4a equation text | arXiv HTML is CC BY 4.0 | Equations independently encoded in `diffgw_reference.xpnr_angles`; attribution retained |
| 2021 predecessor fit procedure | `curve_fit`, sequential refits, fit intervals, and the 40-case polynomial construction are published under CC BY 4.0 | Documents a historical method, not the complete 80-case retune or its outputs |
| BAM 80-waveform single-spin calibration set | Public catalogue: `q={1,2,4,8}`, `chi={0.2,0.4,0.6,0.8}`, tilts `30,60,90,120,150` degrees; [minimal Zenodo record 7673796](https://doi.org/10.5281/zenodo.7673796) and [complete record 7677297](https://doi.org/10.5281/zenodo.7677297), CC BY 4.0. Each record lists 80 `CF21_*.h5` waveform files and no separate fit-target, script, or tensor artifact | Not bundled and not re-fit in this change; the discrete training design is not an accuracy bound |
| `positive` v1.0 fit tool | [Zenodo record 3901856](https://doi.org/10.5281/zenodo.3901856); archived `LICENSE` is MIT while record metadata says `other-open` | Not bundled. Tool availability does not reconstruct the unpublished fit run or license its output |
| Author-controlled PhenomXCP workflow | [Pinned repository](https://github.com/llondon6/PhenomXCP/tree/a411e5966a73445efad0db9dc7806d2195374ab1) includes a historical carrier-tuning workflow and partial artifacts. The root is MIT, while generated `parameter_space_fits.c` says GPL-2.0-or-later and data artifacts have no separate notice | Not bundled. It contains no current angle tensors and no authoritative mapping/version from its partial carrier outputs to final XO4a/XPNR `u_k`; the root license is not treated as blanket output permission |
| XHM v2 author supplement | arXiv:2001.10914v2 source SHA-256 `cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e` contains paper-era amplitude and phase notebooks plus 37 amplitude and 33 phase `.m` tables for modes 21/32/33/44 | Establishes the paper-era fit assets only. No archive manifest maps them to default release `122022`, its all-new amplitude option, or a complete current-release node provider, so no production XHM/XPHM carrier change is made |
| Numeric angle-fit and carrier-deviation tables | Not printed in the cited model papers. XO4a Appendix A points to LALSuite commit `08b90494b3a4a7dc966ca108e6c98eaa3f6e18a7`; LALSuite is GPL-2.0-or-later | No LAL source was inspected, copied, translated, or fitted. Pinned binaries are comparison oracles only |
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
The arrays first appear in DiffGW's initial repository commit,
`ceb6ae15c00bbbd3eaca2bdc12bb3307b38e9beb`; there is no earlier repository
history from which to establish their origin or terms.

## Evidence matrix

| Evidence class | Result | What it establishes |
| --- | --- | --- |
| Paper fidelity | Partial pass | Map, polynomial, guarded transforms, MR ansatzes, standalone Final Twist Eqs. (41)–(45) and XPNR Eqs. (17)–(21), carrier-deformation equation, effective ringdown equation, and HM map have scalar encodings; public angle composition, beta bound, and carrier/asymmetry composition remain open |
| Scalar-reference agreement | Pass for scoped components | Focused published-form cases cover fit values, transforms, ansatz values/derivatives, terminal-state geometry, beta preprocessing, map, window interpretation, HM joins and reductions |
| Torch/JAX parity | Pass for scoped components | Both backends agree with scalar values and derivatives; a dense JAX grid is recorded in the qualification JSON |
| Calibration asset provenance | Provisional gate retained | Published equations, a partial historical procedure, raw CC BY inputs, and an MIT fitting tool are distinguished from the unsupported inherited tensors; a two-case manifest test pins that classification and the unchanged payload |
| External parity | Not qualified | The four optional LAL/PyCBC cases are black-box acceptance gates only; their `0.99` threshold is not an observed lower bound or domain claim |
| Physical validation | Not reproduced | The papers report BAM/NRSur comparisons; this change does not compare DiffGW waveforms with those NR data |
| Full waveform | Fail | Missing published components plus inherited XHM's recorded strict catalog `0/1` and holdout `0/1` failures prevent promotion |

Machine-readable outcomes, versions, commands, and still-failed gates are in
[`imrphenomxpnr-qualification.json`](imrphenomxpnr-qualification.json).

## Promotion gates

Promotion requires all of the following without relaxing existing thresholds:

1. Obtain an independently licensed coefficient release or written
   redistribution permission; alternatively, perform a new fit from the cited
   CC BY data with a fully versioned pipeline, published per-simulation targets,
   sampling/weighting/initialization and sparse-basis choices, uncertainties,
   and holdouts.  Public availability of papers, NR inputs, fitting software,
   or a third-party implementation does not satisfy this gate.
2. Establish an independently specified and licensed PN evolution and explicit
   terminal sampling rule; return terminal `S1`, `S2`, and `Lhat`, demonstrate
   integration convergence and stop-event stability, convert the spins to
   `chi_i = S_i/m_i^2`, then feed that state into Eqs. (3)–(6).
3. Source the taper inputs and connection frequency, compose the checked
   alpha/beta attachments and exterior continuations, then implement the beta
   bound and minimal-rotation convention with value and derivative checks at
   every join.
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
