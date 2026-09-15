# Independent TEOB merger assembly contract

Date: 2026-09-14. Scope: scalar, zero-spin, zero-tide BBH (2,2) mode.
This is an implementer-facing functional specification from the coordinator's
authorized reference audit. It contains mathematical and sampling requirements,
not reference code or calibration tables. Independently derive all stencils
and solves. Preserve MIT package metadata. No oracle fitting or dispatch change.

The physical definitions are in [Nagar et al.](https://arxiv.org/html/1806.01772v2),
Sec. II and Appendix F, and [the earlier NQC paper](https://arxiv.org/abs/1406.6913),
Eqs. (96)–(97). Existing clean components supply the basis, published postmerger
ansatz and remnant options described in
[teob-remnant-compatibility-contract.md](teob-remnant-compatibility-contract.md).
Default remains the UIB2016 remnant; hybrid remnant is explicit opt-in only.
The sampling/index rules below are audited numerical
requirements; they are not asserted to be printed in those papers.

## Auxiliary grid and explicit-target NQC

Inputs are increasing accepted times `t`, canonical states `(r, phi, pr*, j)`,
mode amplitude and unwrapped phase in `h22=A exp(-i phase)`, sampled orbital
frequency and conservative radial acceleration, symmetric mass ratio, and the
fixed continuation step. Sampled derived quantities must be interpolated as
series, not recomputed from interpolated states.

1. Set the cut start to the last accepted knot at or below `t[-1]-20M`
   (clamped to the first knot). Exclude the final accepted knot from the cut.
   Set auxiliary spacing to `min(0.5M, fixed_step)` and use
   `floor((cut_last-cut_first)/spacing)+1` nodes starting at `cut_first`.
   Natural cubic splines separately resample amplitude, unwrapped phase,
   state components, orbital frequency and conservative radial acceleration.
   Reject insufficient or nonfinite support explicitly.
2. On this uniform grid define `D` as the derivative of the degree-four
   interpolating polynomial through five neighboring knots. Use centered
   knots internally and the nearest five at either end. Derive weights from
   polynomial interpolation. Repeated derivatives mean `D(D(f))`.
3. Use the existing clean basis `q=((pr*/(r Omega))^2, rddot/(r Omega^2))`
   and `p=(pr*/(r Omega), pr* r Omega)`. Normalize the amplitude for the NQC
   targets as `A_RW=A/(nu sqrt(24))`. The phase frequency is `D(phase)`.
4. The earliest maximum sampled orbital frequency defines `t_peak_grid`.
   For this zero-spin scope set `t_NQC=t_peak_grid-1M`. The matching index is
   `searchsorted(t_aux,t_NQC,right)-2`. Reject an index outside valid support.
5. The caller MUST supply four targets `(A_t, Adot_t, omega_t, omegadot_t)`
   and a descriptive provenance label. Do not silently select or alter a fit.
   A newly verified primary source resolves the target family for `1<=q<4`:
   [Nagar, Riemenschneider and Pratten (2017)](https://arxiv.org/pdf/1703.06814),
   Appendix A, Table II for `q=1`, and Table IV with Eq. (A3) for `q>1`.
   [Nagar et al. (2018)](https://arxiv.org/pdf/1806.01772v2), Sec. II.B.4,
   restricts replacement by the newer fits to `q>=4`. An explicitly named
   2017 provider may therefore cover `0.16<nu<=0.25`, with the paper's
   printed coefficients and normalization independently verified. Preserve
   the existing 2018 global provider as a distinct published model. Do not
   substitute source-only coefficients or change signs using oracle output.
6. At the selected index independently solve
   `[A_RW*q; D(A_RW*q)] a = [A_t-A_RW; Adot_t-D(A_RW)]` and
   `[D(p); D(D(p))] b = [D(phase)-omega_t; D(D(phase))-omegadot_t]`.
   In particular, differentiate the sampled amplitude-basis PRODUCT. A
   continuous product rule is not equivalent under the specified stencil.
   Report matrices, condition estimates and residuals; reject singular
   systems without regularization or coefficient clipping.
7. Apply `A_corrected=A*(1+q.a)` and `phase_corrected=phase-p.b` to both
   the original and auxiliary samples using each grid's own sampled basis.
   Return both series and all matching metadata. The auxiliary amplitude
   should stay signed if the correction factor becomes negative; expose
   that condition, never silently take an absolute value.

Tests must independently reproduce polynomial derivatives at boundaries,
the discrete osculation conditions (including a product for which the
discrete product rule differs), exact/off-knot matching indices, supplied
target provenance, singular failures and gradients. No full waveform parity
claim follows from solving the matching conditions.

## Ringdown and sampling assembly

This separate component accepts corrected full and auxiliary amplitude/phase
series, auxiliary orbital frequency, mass ratio and final mass/spin. It must
not infer NQC targets or evolve a trajectory.

1. Joining uses the original samples strictly before the auxiliary start and
   auxiliary samples beginning at its SECOND knot. This deliberately omits
   the common first knot, an audited observable join convention. Preserve
   matching time/state series when returning joined metadata.
2. Obtain the orbital maximum from the uniform auxiliary series. For the
   normal single-peak case, use the earliest maximum; reject unsupported
   multiple/edge cases explicitly. Form the degree-six polynomial through
   seven knots centered there. If fewer than three right-hand neighbors
   exist, extend with the last two samples' linear trend. Require three
   left-hand neighbors. The continuous peak estimate is ONE Newton step
   `t0-P'(t0)/P''(t0)`, not a full polynomial-root solve. When `P''(t0)=0`,
   retain `t0`. Report the selected knot and the two derivatives.
3. In total-mass time units, `t_merger=t_peak-3M` and
   `t_match=t_merger+2M`. For fidelity at exact knots perform the attachment
   comparison in final-mass units: compare `t/final_mass` with the threshold
   evaluated as `t_peak/final_mass - 3/final_mass + 2/final_mass` in that
   order; select the last strictly earlier knot, then one knot before it.
   Reject an invalid index.
4. Extend the joined waveform with 1500 samples at the auxiliary spacing.
   Evaluate the existing published (2,2) postmerger ansatz with
   `tau=(t-t_merger)/final_mass`. Its phase exponential uses the published
   `c2phi=alpha21`, while the amplitude uses `c2A=alpha21/2`.
   Anchor phase to the corrected mode at the selected attachment knot.
   Do not rescale amplitude or optimize time/phase. Replace from that knot
   through the end. The standalone ansatz already includes `nu`.
5. An optional uniform output grid starts at the first joined time with
   `floor((last-first)/output_step)+1` samples. Natural cubic interpolation
   acts separately on amplitude and phase before complex reconstruction.
   Preserve full support and return epoch/event information explicitly.

Use synthetic polynomial peaks and an independent phase-anchor calculation
to test these operations before a physical diagnostic. Verify join support,
fixed extension length, exact/off-grid comparisons, time-translation behavior,
no amplitude renormalization, asymptotic frequency, finite gradients and
unsupported input failures. This component stays separate from production
routing until complete public waveform validation is available.
