# XP/XPHM precession boundary qualification (2026-09-25)

## 2026-09-27 isolated XP version/remnant blocker control

A direct public-binary control on `len` used Python 3.12.3, NumPy 2.3.5,
Torch 2.14.0+cpu, JAX 0.11.2, and LALSuite 7.26.15. All waveforms used
`IMRPhenomXP`, convention 1, `phic=0.3`, `phi0=0`, distance 150 Mpc, an
unshifted 0.25-Hz grid, and the same physical inputs for versions 102 and
223 and remnant modes 0 and 3. The equal-mass fixture is the 25/25-Msun
two-spin case below (`f_ref=20` Hz); the near-aligned fixture is the
36/22-Msun up-down case below (`f_ref=40` Hz). The public LALSimulation
binary API supplied only numerical output; no LAL source, coefficient, or
calibration asset was copied. The frozen gates remain relative complex L2
`1e-6` and flat mismatch `1e-8` per polarization.

| Case and band | Native 223/0 vs binary 223/0 L2, plus / cross | Binary 223/0 vs binary 102/0 L2, plus / cross | Native 102/0 vs binary 102/0 L2, plus / cross |
| --- | ---: | ---: | ---: |
| Equal mass, 20–512 Hz | `0.029705677753180615 / 0.03589736628835022` | `0.031765223962125326 / 0.03869141740536079` | `0.0035442845453761617 / 0.004547643714867289` |
| Near aligned, 25–55 Hz | `0.023737179573462445 / 0.02373672163766733` | `0.023737302747141316 / 0.023736880404491217` | `2.965042067384981e-11 / 3.821943469355585e-11` |
| Near aligned, 25–300 Hz | `0.09654477245767022 / 0.09654736105223909` | `0.09654486477183226 / 0.09654748007284461` | `3.2265166254984385e-11 / 4.158838600878678e-11` |

Native version 223/mode 0 and version 102/mode 0 are bit-identical in each
of these three cases: the native 223 path selected NNLO angles. At the
near-aligned fixture, native 223/mode 3 is also bit-identical to native
102/mode 0. Changing the binary remnant from mode 0 to mode 3 changes the
near-aligned full-waveform L2 by at most `2.16e-10` across the larger band;
the failure is an angle-route discrepancy. The binary emits a waveform for
the equal-mass 223/mode-3 request, while both native frontends explicitly
reject its invalid nonaligned MSA averaged-remnant root. That domain-policy
gap has no numerical native residual and remains a release failure.

The version-102 equal-mass residual is **not** specific to exact mass
equality: changing the primary mass to 25.00001 Msun gives native/binary
102/mode-0 L2 `0.0035434142280056907 / 0.004546745513710817`; changing
it to 35 Msun gives `0.04924902660620749 / 0.06218936180015843`.
Aligning only spin 2 at equal mass leaves `0.0036713638518117322 /
0.004648490188983955`. For the near-aligned inputs, the same v102 route
passes at approximately `3–4e-11`. Thus this additional residual lies in a
finite-precession NNLO full-waveform path, not the equal-mass MSA root or
remnant selection. It has not been localized to a particular published NNLO
coefficient or projection equation; no speculative correction is made.

[Pratten et al., Sec. IV.1, Eqs. (17)–(22), and Sec. IV.2](https://arxiv.org/html/2004.06503v2)
distinguish the single-spin NNLO construction from two-spin MSA, identify
singular equal-mass MSA couplings, and describe the NNLO angle fallback for
unstable initialization. [Chatziioannou et al., Appendix E](https://arxiv.org/pdf/1703.03967v3)
notes the near-aligned root/PN conditioning and proposes a dedicated
small-misalignment expansion as an improvement. The independent scalar cubic
receipt in `docs/waveforms/research/xp-v223-reference-closure.md` finds the
reference inverse-cosine endpoint only about `3.30e-37` from its limit for
the near-aligned `1e-10` perturbation, far below float64 resolution.
[Gerosa et al., Sec. IV.2](https://arxiv.org/html/2304.04801) give a
regularized variable for a different precession formulation but leave a
formal zero-amplitude resonance expansion to future work. These sources do
not specify a complete v223 float64 root, phase, remnant, and waveform-chart
continuation that would reproduce the pinned binary at these boundaries.
The output cannot justify copying a hidden binary fallback or fitting a
correction. **The XP release verdict remains fail**, with the frozen gates
unchanged.

The unchanged `scripts/qualify_xp_xphm_boundaries.py` driver was rerun on
`len` with `PYTHONPATH=src`, `OPENBLAS_NUM_THREADS=1`, and
`OMP_NUM_THREADS=1`. Its public-binary rows reproduce the equal-mass
mode-0 L2 above, pass both near-aligned v102 modes at at most `4.16e-11`,
fail both near-aligned v223 modes at `0.0237` (25–55 Hz) and `0.0965`
(25–300 Hz), and pass generic nonzero-`phi0` at at most `1.15e-13`.
The equal-mass mode-3 row records the explicit native domain rejection.
The XPHM diagnostic and holdout remain `SOVEREIGN_NR_CALIBRATED` rather
than qualified (`0.226–0.556` per-polarization L2). The focused boundary,
scalar-reference, and source-chart test modules passed `16/16` on `len`.

## 2026-09-26 generic nonzero-`phi0` full-waveform closure

[Pratten et al., Appendix C Eqs. (67)--(68), (76)--(85), Appendix B Eq. (63),
Appendix E Eqs. (110)--(113), and Appendix F Table IV](https://arxiv.org/html/2004.06503v2)
define the source observer azimuth, J-frame Euler and polarization geometry,
quadrupole twist-up, and convention choices. From those published frames, we
**derive** the public-binary coordinate map: rotate both in-plane input spins
by `Rz(-phi0)`, pass `phiRef=phic` to the binary, and multiply its completed
quadrupole polarizations by `exp(-2i phi0)` for conventions 0/1/6. The rotation
and completed-strain factor are our frame derivation, not equations quoted
from the paper or coefficients obtained from LAL source.

The independent mpmath Appendix-C scalar chart and Torch/JAX full waveforms
now check this covariance. On `len` with Python 3.12.3, NumPy 2.3.5,
Torch 2.14.0+cpu, JAX 0.11.2, and public LALSuite 7.26.15,
`IMRPhenomXP`, `prec_version=223`, `convention=1`, `final_spin_mod=3`,
`(m1,m2)=(35,25) Msun`, spins `(0.2,0.1,0.3)` and `(-0.1,0.05,-0.2)`,
`phic=0.3`, `phi0=0.4`, and the direct unshifted 20--512 Hz grid at 0.25 Hz,
the mapped binary comparison gives:

| Convention | Plus/cross complex relative L2 | Largest flat mismatch | Verdict |
| --- | --- | ---: | --- |
| 0 | `1.1410970098602045e-13 / 1.1213128890693516e-13` | 0 | pass |
| 1 | `1.1447907380982857e-13 / 1.1248152247989706e-13` | 0 | pass |
| 6 | `1.1343201250715765e-13 / 1.1143297006383575e-13` | `9.992007221626409e-16` | pass |

All pass the unchanged `1e-6` and `1e-8` gates. These measurements qualify
this declared generic nonzero-azimuth fixture; they do not establish a wider
precession-domain guarantee. The prior `NOT_EVALUABLE` statements below are
historical and superseded by this derived-map check.

The overall XP/XPHM release verdict remains **fail**: equal-mass mode 3 is an
unsupported MSA-remnant root, near-aligned v223 binary residuals remain above
the frozen gate, and the XPHM full-waveform cases remain failing.

## 2026-09-26 independent source-frame transport gate

[Pratten et al., Appendix C, Eqs. (67)--(83)](https://arxiv.org/html/2004.06503)
define the source observer vector, the J frame, and its Euler/polarization
angles. A new high-precision scalar reference, `source_chart_transport` in
`src/diffgw_reference/xp_msa_generic_boundary.py`, evaluates the relative
regular-chart transport at fixed intrinsic J. Equation (79) gives
`delta_alpha = alpha0(phiRef)-alpha0(0)` and
`delta_gamma = -(phi_JL0(phiRef)-phi_JL0(0))`, represented modulo `2*pi`.
The destination chart supplies `theta_JN` and `zeta`. Because J is fixed when
only the observer azimuth changes, its regular-chart `phi_JL0` is fixed and
`delta_gamma` is zero. At exactly aligned J, `phi_JL0` is undefined; the
scalar reference rejects Euler transport instead of choosing a gauge.

`tests/test_xp_source_chart_transport_math_reference.py` compares this scalar
chart and transport with Torch and JAX v223 geometry at four nonzero observer
azimuths, including `phic-phi0 = -0.1` and a phase-wrap case. The orbital-L
value is supplied by the separately identified v223 compatibility expression;
the test establishes the **observer-frame transport**, not independent
provenance for that compatibility coefficient. On `len`, the new tests plus
the existing source-frame/composition/generic-boundary reference files passed
`26/26` with Torch 2.14.0+cpu, JAX 0.11.2, and pytest 9.1.1.

This result checked the internal source chart only. Its then-unresolved
public-binary mapping and production-routing statements were superseded by
the generic nonzero-`phi0` closure above; frozen tolerances did not change.

## 2026-09-26 frozen angle-version control

[Pratten et al., Appendix F, Tables III and V](https://arxiv.org/html/2004.06503)
distinguish NNLO angle version 102 from MSA version 223 (with NNLO fallback
when MSA initialization fails), and specify the remnant choice separately.
At integrated PR #42 source head `b4c6e0d`, the qualification driver now
evaluates both versions and remnant modes 0/3 at identical near-aligned
physical inputs. It retains the prior v223/mode-3 case IDs. The public
LALSimulation binary API remains a numerical oracle only;
no implementation or coefficient is copied or fitted from it.

On `len`, Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, NumPy 2.3.5,
SciPy 1.16.3, PyCBC 2.11.0, and LALSuite 7.26.15, the full complex
polarization comparison gave the following. Each L2 cell is `h+ / hx`;
mode-0 and mode-3 values are shown as ranges because they are nearly equal.
The unchanged strict gates are complex relative L2 `1e-6` and flat mismatch
`1e-8` for each polarization.

| Band (Hz) | Angle version | Modes 0/3 complex relative L2, `h+ / hx` | Largest mismatch | Strict verdict |
| --- | --- | --- | ---: | --- |
| 25–55 | 102 | `1.87–2.97e-11 / 2.41–3.82e-11` | `2.22e-16` | both pass |
| 25–55 | 223 | `0.02373718 / 0.02373672` | `2.82e-4` | both fail |
| 25–300 | 102 | `1.87–3.23e-11 / 2.41–4.16e-11` | `1.11e-16` | both pass |
| 25–300 | 223 | `0.09654477 / 0.09654736` | `4.66e-3` | both fail |

This control shows that the public-binary disagreement is version-specific
at these fixed inputs and is not removed by changing remnant mode. It does
not establish which convention should define the physical aligned endpoint
or justify changing production waveform equations. The existing
`test_xp_v223_azimuth_boundary_diagnostic.py` and
`test_xp_aligned_boundary_clean.py` encode those distinct contracts.
These two test files passed `27/27` on `len` in 6.50 seconds in the isolated
PyCBC environment. The shared validation environment was restored to
NumPy 2.5.3 and SciPy 1.18.1 without PyCBC.
At this angle-version control stage, generic observer `phi0` remained
**not evaluable** against the public binary. The frame-map derivation above
supersedes that status. Overall release qualification remains **fail**.

## 2026-09-26 explicit-remnant policy amendment

The earlier receipt below combined the paper-prescribed NNLO *angle* fallback
with an implicit change from `final_spin_mod=3` to mode 0. That remnant
substitution is superseded. The current public XP/XPHM policy keeps these two
choices independent:

- MSA v223 still falls back to the paper-sourced NNLO angles when MSA
  initialization is invalid.
- Explicit `final_spin_mod=3` remains a request for the MSA averaged remnant.
  A nonaligned equal-mass row or other genuinely invalid MSA remnant root
  raises the same clear diagnostic in Torch and JAX; it is never silently
  evaluated with mode 0.
- Explicit `final_spin_mod=0` remains finite and continuous across the tested
  equal/near-equal boundary and retains differentiability in both frontends.
  Machine-level transverse-spin degeneracies continue to use the analytic
  aligned remnant limit.

JAX eager calls made directly to the remnant primitive raise `ValueError`.
Public XP evaluation uses `lax.cond`; consequently, invalid eager public calls
and invalid JIT executions cross a compiled callback boundary and surface a
`jax.errors.JaxRuntimeError` from the compiled XLA host callback that contains
the same diagnostic rather than preserving the Python exception class. Valid
JIT execution and autodiff remain supported.

On `len`, the focused policy suite
(`test_xp_msa_remnant_boundary_policy.py`, `test_jax_public_xp_generic_v223.py`,
and `test_xp_msa_production_clean.py`) passed `37/37` in 385.06 seconds with
Torch 2.14.0+cpu, JAX/JAXlib 0.11.2, NumPy 2.5.3, SciPy 1.18.1, and pytest
9.1.1. It includes direct and compiled invalid-root checks, mode-0 boundary
continuity, gradients, frontend parity, and valid batched JIT coverage.

This is a boundary-policy correction, not new external qualification. The
strict full-complex-polarization gates are unchanged. The earlier equal-mass
numerical row below was evaluated under the superseded implicit mode-0
substitution; it is historical evidence, not a current mode-3 residual.
The overall release verdict remains **fail**.

### Integrated-head rerun on `len`

The driver at this historical integrated head explicitly evaluated both
equal-mass remnant choices. At integrated source head `2e654a3` (the later
`ccc3588` commit changed documentation only), Python 3.12.3, Torch 2.14.0+cpu,
JAX 0.11.2, NumPy 2.5.3, and the installed LALSuite 7.26.15 produced:

| XP equal-mass choice | Torch/JAX direct complex relative L2, plus / cross | Independent binary-oracle result |
| --- | ---: | --- |
| Default `final_spin_mod=3` | Both reject the invalid nonaligned MSA remnant root with the same diagnostic; public JAX surfaces it through `JaxRuntimeError` | Generation error; no numerical residual exists |
| Explicit `final_spin_mod=0` | `1.9282897530721765e-14 / 1.9297675234635243e-14`; both polarizations pass parity | Plus `0.02970567775318068 / 0.0004410058872005296`; cross `0.035897366288350226 / 0.0006443233349125999` (L2 / mismatch); both fail |

The same integrated run retains the near-aligned external relative L2 values
`0.023737179603638348` and `0.09654477266816827` for plus, and the XPHM
diagnostic/holdout plus values `0.22881790727328782` and
`0.5041125739427325`. Their strict external verdicts remain **fail**. Generic
nonzero observer `phi0` then lacked an independently derived public-binary
parameter mapping, so its external comparison was **not evaluable**. The
integrated-head focused policy suite passed `37/37` in 388.54 seconds.

## Scope and evidence classes

This receipt audits draft PR #42 at
`4814835f7744fe00a5863ed70eb590d41a7e3ac3` and qualifies only generic
equal mass, generic nonzero observer `phi0`, near-aligned/instability routing,
and the XPHM precession boundary. The unchanged strict full-complex-polarization
gate is relative L2 `1e-6` and flat mismatch `1e-8`.

The PR's nine GitHub checks were green and its existing validation text
correctly left these boundaries open. Those checks establish integration, not
release qualification. The results below keep four evidence classes separate:

1. **Math fidelity:** paper-derived equations or the paper-prescribed NNLO
   fallback, without LAL source or fitted oracle coefficients.
2. **Torch/JAX parity:** direct, unshifted complex `h+` and `hx` comparisons.
3. **External validation:** the installed LALSimulation public binary API used
   only as a numerical oracle.
4. **Release qualification:** all applicable classes must pass.

The independent run used host `len`, Python 3.12.3, NumPy 2.5.3, SciPy
1.18.1, Torch 2.14.0+cu130 (CPU forced), JAX/JAXlib 0.11.2 (CPU FP64),
pytest 9.1.1, and the `lalsuite` distribution 7.26.15. The executable receipt
is `scripts/qualify_xp_xphm_boundaries.py`.

The focused strict pytest run covered the public XP generic route, independent
generic-boundary math reference, promoted near-aligned fallback gates, XPHM
rotation reference, and clean-room XPHM NNLO composition: `67 passed` in
496.14 seconds, with zero skips, xfails, or deselections.

## Historical correction at the prior qualification head

The published generic MSA parameterization is singular at equal mass and its
distinct-root solution is invalid when the initialization roots coalesce.
The production adapters at that head reused the existing MSA initialization-validity
result before selecting the clean distinct-root kernel. Invalid rows take the
established NNLO angle fallback. Equal-mass generic rows also select the NNLO
remnant because the MSA averaged-remnant expression shares the equal-mass
singularity. The 2026-09-26 amendment above supersedes only that silent
remnant-mode substitution. No continuation, new threshold, LAL-derived
expression, or fitted coefficient is introduced.

## Historical pre-amendment native residuals

Each cell is `relative L2 / mismatch` for `(plus, cross)` respectively.

| Boundary | Torch vs JAX | Boundary behavior | Verdict |
| --- | --- | --- | --- |
| Generic equal mass | `(4.5485282301920294e-14 / 0, 4.542345628206036e-14 / 0)` | Torch vs forced NNLO fallback `(0 / 0, 0 / 0)`; JAX vs fallback `(3.09319461515901e-14 / 0, 3.088098683457194e-14 / 5.551115123125783e-16)` | parity/fallback pass |
| Generic nonzero `phi0` | `(2.1998567136506545e-14 / 0, 2.1580647978125144e-14 / 0)` | Torch vs inherited route `(0 / 0, 0 / 4.440892098500626e-16)`; JAX vs inherited route `(1.224565931624615e-14 / 0, 1.2339688469043134e-14 / 0)` | parity pass only |
| Near aligned, 25--55 Hz | `(1.1928598420584253e-14 / 2.220446049250313e-16, 1.1931001987773128e-14 / 0)` | tiny vs exact aligned `(3.0602795845805276e-11 / 0, 3.9445611075509205e-11 / 2.220446049250313e-16)` | parity/continuity pass |
| Near aligned, 25--300 Hz | `(3.201086826590522e-14 / 0, 3.20117733545435e-14 / 0)` | tiny vs exact aligned `(3.370259364455347e-11 / 0, 4.344117178177736e-11 / 0)` | parity/continuity pass |

At the audited PR head, equal-mass Torch/JAX relative L2 was
`0.15536521149740812` plus and `0.15342033548414735` cross. The near-aligned
JAX output was nonfinite, while the Torch tiny-vs-aligned L2 was approximately
`0.0245` before the instability and `0.13` across it. The correction removes
those frontend and boundary discontinuities by selecting the declared fallback.

## Historical pre-amendment independent full-waveform residuals

| Boundary/case | Plus L2 / mismatch | Cross L2 / mismatch | External verdict |
| --- | ---: | ---: | --- |
| XP generic equal mass | `0.09623529299354046 / 0.004632042422924254` | `0.09462201778421948 / 0.004478172581788398` | fail |
| XP near aligned, 25--55 Hz | `0.023737179603638348 / 0.0002817282964121537` | `0.02373672166784322 / 0.00028171783997021915` | fail |
| XP near aligned, 25--300 Hz | `0.09654477266816827 / 0.004660451020372602` | `0.09654736126273701 / 0.004660702227981206` | fail |
| XPHM diagnostic | `0.22881790727328782 / 0.025734053284783842` | `0.22641701045075546 / 0.025321404029660988` | fail |
| XPHM holdout | `0.5041125739427325 / 0.11210914391477245` | `0.5560212744607049 / 0.1341335900553805` | fail |

Generic nonzero observer `phi0` is not externally evaluable: there is no
independently derived mapping from this public diffGW parameter to the public
LAL binary parameters. Native agreement therefore cannot promote it.

## Historical boundary verdicts before frame-map closure

| Boundary | Math fidelity | Torch/JAX | External | Release |
| --- | --- | --- | --- | --- |
| Generic equal mass | mode 0 passes as explicit NNLO fallback; explicit mode 3 is an MSA-remnant domain error; no MSA continuation claimed | same policy/diagnostic across frontends | fail | **fail** |
| Generic nonzero `phi0` | source-frame transport remains unclosed | pass on inherited route | not evaluable | **fail** |
| Near aligned before instability | pass as invalid-root NNLO fallback | pass | fail | **fail** |
| Near aligned across instability | pass as invalid-root NNLO fallback | pass | fail | **fail** |
| XPHM precession layer | paper/scalar rotation gates pass | pass in focused composition tests | full waveform fails | **fail** |

This table records the pre-closure state of the generic `phi0` row. The
current mapped-binary result above passes for its declared fixture. The
overall release gate remains **fail** because the XP fallback boundaries and
XPHM's XHM carrier dependency still disagree with the public binary.
