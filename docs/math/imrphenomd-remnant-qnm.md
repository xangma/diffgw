# IMRPhenomD remnant and QNM inputs

## Scope and status

`imrphenomd-remnant-qnm-paper-precision/v1` specifies the remnant fits from
[Husa et al., Paper I, arXiv:1508.07250v2](https://arxiv.org/html/1508.07250v2)
and the conversion of Kerr QNM frequencies to the binary's total-mass units.
The executable expressions are in
[`src/diffgw_reference/imrphenomd_remnant.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomd_remnant.py).
This extends the [amplitude specification](imrphenomd-amplitude.md), which
previously took ringdown inputs as supplied values. The Kerr eigenfrequencies
are **numerical inputs**: Paper I interpolates pre-existing QNM data, but does
not publish an analytic function or the spline used by DiffGW.

## Inputs, conventions, and options

For aligned component spins $\chi_i$, $M=m_1+m_2$, and
$\eta=m_1m_2/M^2$, Section III.1 defines

$$S=\frac{m_1^2\chi_1+m_2^2\chi_2}{M^2},\qquad
\hat S=\frac{S}{1-2\eta}.$$

## Mathematics and components

The printed Eq. (22) final-spin fit is

$$\begin{aligned}
a_f={}&S+2\sqrt3\eta-4.399\eta^2+9.397\eta^3-13.181\eta^4\\
&+\eta(-0.085S+0.102S^2-1.355S^3-0.868S^4)\\
&+\eta^2(-5.837S-2.097S^2+4.109S^3+2.064S^4).
\end{aligned}$$

Eqs. (23)–(24) give $e=E_{\rm rad}/M$:

$$e_{\rm NS}=0.0559745\eta+0.580951\eta^2-0.960673\eta^3
+3.35241\eta^4,$$

$$e=e_{\rm NS}\frac{1+\hat S(-0.00303023-2.00661\eta+7.70506\eta^2)}
{1+\hat S(-0.67144-1.47569\eta+7.30468\eta^2)},\qquad M_f=M(1-e).$$

If the Kerr solver supplies a *cyclic* ordinate $M_f f_{220}(a_f)$, Eq. (17)
and the unit convention imply $M f_{220}=(M/M_f)(M_f f_{220})$. The same
scaling applies to the positive damping frequency. Angular eigenfrequency
$\omega$ must first be divided by $2\pi$; its negative imaginary part becomes
positive damping. The output frequencies are dimensionless in the **initial
total mass**, as the amplitude branch requires.

| Component ID | Evidence | Torch/JAX check |
| --- | --- | --- |
| `phenomd.remnant.coordinates.v1` | Paper I, Section III.1 | Mass/spin substitutions and limiting cases |
| `phenomd.remnant.spin.v1` | Eq. (22) | SymPy printed fit and explicitly bound inherited coefficients; selected values and spin derivative |
| `phenomd.remnant.energy.v1` | Eqs. (23)–(24) | Printed fit, inherited-coefficient algebra, selected values and spin derivative |
| `phenomd.remnant.mass.v1` | Eq. (24), mass conservation | $M_f$ values and derivative |
| `phenomd.qnm.total_mass_scaling.v1` | Eq. (17), angular-to-cyclic conversion | Both QNM channels and composite spin derivative |
| `phenomd.qnm.natural_cubic_compatibility.v1` | DiffGW numerical policy, not a paper equation | Independent SciPy natural spline of supplied knots, plus stricter-solver holdouts |

## Implementation mapping

`COMPONENT_BINDINGS` in the executable specification maps each ID to its
reference function, Torch and JAX entry points, and maintained test names. A
structural test checks that every binding and source resolves.

## Evidence and limitations

The SymPy module imports no production waveform, coefficient, or QNM module.
Its default coefficients contain **only printed digits**. Torch/JAX use
additional inherited digits in `diffgw_core.fits`; the tests supply those
digits explicitly to the same equations to isolate algebra verification from
coefficient provenance. The inherited $\eta S^2$ coefficient is
`0.1014665242971878`, while Eq. (22) prints `0.102`: those are not ordinary
roundings at the printed precision (`0.101466...` rounds to `0.101`). The
inherited value matches the [29 September 2015 LALSuite update by paper
coauthor Sebastian Khan](https://github.com/lscsoft/lalsuite-archive/commit/cf74eff8e3076bae8daf64e98d82cddbe5d59abb),
whose commit message explicitly says it updated the final mass and spin
formulas. This is the earliest public occurrence of the exact value located in
this audit; the [paper's TeX source](https://arxiv.org/src/1508.07250v2)
contains only `0.102` for this term. DiffGW adopted the longer value in commit
`4fd62c4`, replacing an earlier `0.102` implementation. Both the [first and second versions of Paper
I](https://arxiv.org/abs/1508.07250) predate that update and print `0.102`.
This establishes a match to that historical implementation, not an independent
or MIT-compatible source for DiffGW's extra digits. The public record does not
establish whether the discrepancy is a paper typo or a small post-paper fit
revision; no separately released author fit record or license for these extra
digits was located. The reference therefore keeps the **published** and **historical
implementation** coefficient sets separate; paper-level agreement is qualified
to printed-fit tolerances.

For validation, the default symbolic expression is the published Eq. (22).
The historical value is bound explicitly in the backend comparison; it is a
compatibility choice with unresolved fit provenance, not an independently
recovered paper coefficient. A maintained audit checks every printed remnant
fit digit: only the $\eta S^2$ final-spin coefficient crosses a rounding
midpoint. Holding all other coefficients fixed, the two choices differ by
$\Delta a_f=(0.1014665242971878-0.102)\eta S^2$. Since
$|S|\leq1-2\eta$ for physical aligned spins and $0<\eta\leq1/4$,
$|\Delta a_f|\leq(2/27)|\Delta b_2|<4.0\times10^{-5}$ over that full
domain. This bound concerns the one coefficient only; differences from all
other extra implementation digits are assessed separately.

### Independent final-spin recalibration

The maintained [refit script](https://github.com/xangma/diffgw/blob/main/scripts/refit_imrphenomd_final_spin.py)
implements the four free $\eta S^k$ terms ($k=1,\ldots,4$) identified after
Eq. (21) of [Paper I](https://arxiv.org/html/1508.07250v2). It fixes every
other coefficient at the *printed* Eq. (22) precision and solves the resulting
linear least-squares problem for $b_1,\ldots,b_4$. An optional one-parameter
fit holds $b_1,b_3,b_4$ at their printed values to expose covariance between
terms. The input is the [427-case NR table](https://arxiv.org/src/1611.00332v2/anc/FinalStateUIB2016_suppl_NRdata.txt)
from the later [Jiménez-Forteza et al. study](https://arxiv.org/html/1611.00332v2),
with SHA-256
`3a20968a77b2292754404c9cefcaab373f14fc3a47adc1546df1ec442fbea0a3`.
The script checks that hash and imports neither DiffGW waveform code nor LAL.
It does not redistribute the NR table.

Protocol `phenomd-final-spin-four-term-2017/v1`, run on `len` with Python
3.12.7 and NumPy 2.1.1. The RMS column uses uniform case weights for both
fits, regardless of the optimization weights:

| Fit on all 427 cases | $b_2$ with four free terms | $b_2$ alone | In-sample RMS in $a_f$ |
| --- | ---: | ---: | ---: |
| Printed Eq. (22) | 0.102 | — | 0.007196 |
| Uniform least squares | 0.080546 | 0.104558 | 0.006983 |
| Ancillary heuristic weights | 0.118953 | 0.107419 | 0.006999 |

For an out-of-catalog check, four fits each exclude all cases from one of BAM,
GaTech, RIT, or SXS. Their combined held-out RMS is 0.007114, versus 0.007196
for the fixed printed equation on the same cases. Improvement is mixed by
catalog: the refit is better for BAM and SXS but worse for GaTech and RIT.
The four fold-specific $b_2$ estimates span 0.0706–0.0950. The later paper's
own [Table 6](https://arxiv.org/html/1611.00332v2) likewise reports roughly
0.007 RMS for the PhenomD ansatz on these 427 cases, an independent check of
the residual scale rather than of our particular coefficients.

Replacing only the printed $b_2=0.102$ with the historical
`0.1014665242971878` changes predictions for these cases by at most
$3.30\times10^{-5}$, versus a $7.20\times10^{-3}$ RMS fit residual. The
historical coefficient therefore cannot be identified to its extra digits by
this later data and protocol. The original study used 106 cases and a staged
calibration; its full machine-readable calibration values and exact choices
are not provided by Paper I. Rounded fixed coefficients also prevent an exact
reconstruction. **These estimates are an independent recalibration of the
printed ansatz, not a reproduction of the 2015 fit or grounds to replace a
production coefficient.**

The [machine-readable report](data/imrphenomd_final_spin_refit_2017_v1.json)
records the input and script hashes, all four fitted coefficients, source
counts, rank and conditioning, error metrics, and each catalog holdout.
To rerun, download the linked ancillary text file separately, then use
`python scripts/refit_imrphenomd_final_spin.py /path/to/FinalStateUIB2016_suppl_NRdata.txt --comparison-b2 0.1014665242971878 --output report.json`.
Synthetic coefficient recovery and input-hash checks live in
`tests/test_imrphenomd_final_spin_refit.py`.

DiffGW's QNM asset consists of generated Kerr $(2,2,0)$ eigenvalues sampled
on a signed-spin grid, converted to cyclic units, rounded to six significant
digits, and represented by a natural cubic spline. The reference receives the
knots and ordinates explicitly; it does not treat them as a paper equation.
The separate [QNM provenance and generator](../provenance_ledger.md#212-clean-room-kerr-quasinormal-mode-solvers-generation-torchwaveqnm)
and `tests/data/kerr_qnm_220_leaver.json` record stricter eigenproblem checks.
At zero spin, the test also checks both cyclic channels against row 0 of
[Berti's published Schwarzschild $s=\ell=2$ QNM data](https://pages.jh.edu/eberti2/ringdown/s2l2.dat),
converting the author's $2M\omega$ values by $4\pi$.
On selected $|a_f|<0.9$ holdouts, the rounded D representation is required to
stay within $10^{-6}$ absolute **and** $10^{-5}$ relative in both cyclic
channels. All 93 stored high-spin holdouts and stricter solves with
$0.994<|a|\leq0.99999$ require $10^{-8}$ absolute and $10^{-6}$ relative
agreement from both generated continuations, including $a=0.9976$.
Those holdouts share the numerical solver lineage, so they check the asset's
approximation error rather than supply an independent physical calibration.
Two $|a|=0.9999$ QNM-input checks also compare both signed-spin branches with
the $s=-2$, $\ell=2$, $m=\pm2$, $n=0$ angular-frequency rows in
[Lo, Sabani, and Cardoso's generalized Sasaki–Nakamura data](https://the-center-of-gravity.com/data-and-routines/)
([paper](https://arxiv.org/abs/2504.00084)). They convert $M\omega$ to
$M f$ by $2\pi$. This is a different numerical method and sits beyond the
remnant's $|a_f|=0.999$ cap, so it checks the direct QNM input only. The author
archive is not redistributed here. Its $a=0.9976$ damping row disagrees by
about 1.65% with the stricter continued-fraction solve and its immediate
neighboring author rows; we exclude that row from the independent-method
acceptance check rather than treating every published datum as exact.

Both frontends apply the numerical remnant policy
$a_f=\min(0.999,\max(-0.999,a_{f,\mathrm{fit}}))$. Its slope with respect to
an input is zero where the raw fit exceeds the cap; no smooth derivative is
assigned at the clamp threshold. The independent spline reference uses supplied
knots over $[-0.999,0.999]$ and does not itself clamp; the full intrinsic
composition now applies the remnant cap before selecting its QNM input.

The shared rounded QNM interpolation comparison covers $|a_f|\leq0.994$.
Above that threshold, both frontends evaluate the independently generated
degree-5 Kerr spline, convert its angular frequencies to cyclic units, and
retain the remnant cap $|a_f|\leq0.999$. The prior JAX continuation of the
six-significant-digit natural cubic reached about 1.0% relative damping error
against same-lineage high-spin holdouts on the positive-spin side; that
continuation is no longer used there. Endpoint tests check the declared policy
on both signed-spin sides and test the
full remnant call for a high-spin, unclamped binary and a clamped binary. The
$(150,1,0.999,0.999)$ and $(1000,1,1,1)$ binaries are numerical branch probes,
not physical-accuracy claims for those extreme mass ratios. The
generated-root path check shares each frontend's numerical asset, so it
establishes path selection rather than an independent physical QNM result. A
separate eigenproblem holdout checks that asset and cross-frontend accuracy in
the high-spin continuation. The fit's physical accuracy over the full
parameter space is not established here. The
[full intrinsic composition](imrphenomd-intrinsic.md) now checks how these
high-spin inputs propagate through the complete complex waveform; the
[phase reference](imrphenomd-phase.md) covers the intrinsic phase separately.

## Reproduce the checks

Run `pytest -q tests/test_imrphenomd_remnant_math_reference.py`; the maintained
`--require-math-reference` gate also collects this module.
