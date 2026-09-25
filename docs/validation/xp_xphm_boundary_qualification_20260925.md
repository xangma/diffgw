# XP/XPHM precession boundary qualification (2026-09-25)

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

## Adopted correction

The published generic MSA parameterization is singular at equal mass and its
distinct-root solution is invalid when the initialization roots coalesce.
The production adapters now reuse the existing MSA initialization-validity
result before selecting the clean distinct-root kernel. Invalid rows take the
established NNLO angle fallback. Equal-mass generic rows also select the NNLO
remnant because the MSA averaged-remnant expression shares the equal-mass
singularity. No continuation, new threshold, LAL-derived expression, or fitted
coefficient is introduced.

## Exact native residuals

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

## Exact independent full-waveform residuals

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

## Boundary verdicts

| Boundary | Math fidelity | Torch/JAX | External | Release |
| --- | --- | --- | --- | --- |
| Generic equal mass | pass as explicit NNLO fallback; no MSA continuation claimed | pass | fail | **fail** |
| Generic nonzero `phi0` | source-frame transport remains unclosed | pass on inherited route | not evaluable | **fail** |
| Near aligned before instability | pass as invalid-root NNLO fallback | pass | fail | **fail** |
| Near aligned across instability | pass as invalid-root NNLO fallback | pass | fail | **fail** |
| XPHM precession layer | paper/scalar rotation gates pass | pass in focused composition tests | full waveform fails | **fail** |

The declared overall release gate is **fail**. The remaining blockers are the
unvalidated generic `phi0` source-frame mapping and external full-waveform
disagreement for the XP fallback boundaries and XPHM's XHM carrier dependency.
