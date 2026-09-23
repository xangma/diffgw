# IMRPhenomD intrinsic phase and joins

`imrphenomd-phase-joins-paper-precision/v1` is the executable phase
specification in [`imrphenomd_phase.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomd_phase.py).
Its primary source is [Khan et al., Paper II, arXiv:1508.07253v2](https://arxiv.org/html/1508.07253v2),
Eqs. (13)–(16), (27)–(28), (31), (37), Appendix B and Table 5. The SymPy
module imports no production waveform implementation or coefficients.

Let $u=Mf$ be frequency in the initial total mass, $v=(\pi u)^{1/3}$,
$\eta=m_1m_2/M^2$, $\delta=(m_1-m_2)/M$,
$\chi_s=(\chi_1+\chi_2)/2$, and $\chi_a=(\chi_1-\chi_2)/2$.
With $M$ in seconds, the frontend frequency in Hz is $f=u/M$.
The [amplitude specification](imrphenomd-amplitude.md) defines the same
spin coordinates and supplies the amplitude peak; the
[remnant specification](imrphenomd-remnant-qnm.md) supplies the positive
$f_{\rm RD}$ and $f_{\rm damp}$ in total-mass units.

The aligned-spin TaylorF2 phase from Paper II Eq. (27) is

$$\phi_{\rm PN}(u)=\frac{3}{128\eta v^5}
\left[1+c_2v^2+c_3v^3+c_4v^4
+(c_5+c_{5\log}\log v)v^5
+(c_6+c_{6\log}\log(4v))v^6+c_7v^7\right].$$

The exact rational, $\pi$, and Euler-constant coefficients are in
`pn_phase_coefficients`, transcribed from Appendix B Eqs. (46)–(53).
In particular, the logarithmic terms retain the paper's distinct
$\log v$ and $\log(4v)$ bases. The arbitrary coalescence time and phase
constants are excluded from this intrinsic function.

Paper II Eq. (28) adds four calibrated inspiral corrections:

$$\phi_{\rm ins}(u)=\phi_{\rm PN}(u)+\frac{1}{\eta}
\left(\sigma_1u+\frac34\sigma_2u^{4/3}
+\frac35\sigma_3u^{5/3}+\frac12\sigma_4u^2\right).$$

The arbitrary constant $\sigma_0$ is set to zero; the frontend phase
conventions below supply the global phase offset.

The middle and merger-ringdown phase integrals of Eqs. (16) and (14) are

$$\begin{aligned}
\phi_{\rm int}(u)&=\frac1\eta\left(\beta_0+\beta_1u
+\beta_2\log u-\frac{\beta_3}{3u^3}\right),\\
\phi_{\rm MR}(u)&=\frac1\eta\left(\alpha_0+\alpha_1u
-\frac{\alpha_2}{u}+\frac43\alpha_3u^{3/4}
+\alpha_4\arctan\frac{u-\alpha_5f_{\rm RD}}{f_{\rm damp}}\right).
\end{aligned}$$

The joins are at $u_1=0.018$ and $u_2=f_{\rm RD}/2$, in the model
regime where $u_2>u_1$.
The reference solves $\beta_0,\beta_1$ by matching the phase and its
$u$ derivative between inspiral and intermediate at $u_1$, then solves
$\alpha_0,\alpha_1$ the same way at $u_2$ (Eq. (37)). This fixes a
$C^1$ piecewise phase. The printed Table 5 $\beta_1$ and $\alpha_1$
rows are retained as source evidence; the joined branches use the
continuity-determined values instead.

Every fitted row $\Lambda\in\{\sigma_{1:4},\beta_{1:3},\alpha_{1:5}\}$
uses Eq. (31), with $x=\chi_{\rm PN}-1$:

$$\Lambda=c_0+c_1\eta+x(c_2+c_3\eta+c_4\eta^2)
+x^2(c_5+c_6\eta+c_7\eta^2)+x^3(c_8+c_9\eta+c_{10}\eta^2).$$

The default rows in the reference contain only the digits printed in
Table 5. Torch and JAX use inherited extra digits in `diffgw_core.fits`.
Backend checks pass those production rows **explicitly** into the reference
formula. This tests the fit algebra and phase assembly conditional on those
digits; it does not establish that the extra digits were published or
independently calibrated.

For zero $t_c$ and $\phi_c$, the frontends expose two phase conventions.
Their *coalescence* convention returns
$\Psi_c(u)=-\pi/4+\phi(u)-s_{\rm peak}u$.
Their *reference-frequency* convention returns
$\Psi_r(u)=\phi(u)-\phi(u_r)-s_{\rm peak}(u-u_r)$,
where $u_r=Mf_{\rm ref}$. The common slope $s_{\rm peak}$ is the
merger-ringdown derivative of Eq. (13) at the amplitude peak using the
**raw fitted** $\alpha_1$, not the continuity-determined $\alpha_1$.
This global time gauge is a DiffGW frontend convention, recorded separately
from the paper's intrinsic piecewise phase. Both conventions add
$2\pi f t_c-2\phi_c$ when requested.

## Component traceability

| Component IDs | Paper evidence | Executable reference | Maintained frontend check |
| --- | --- | --- | --- |
| `phenomd.phase.fit.{sigma1–4,beta1–3,alpha1–5}.v1` | Eq. (31), Table 5 | `fit_expr`, printed rows | Row transcription and Torch/JAX `eval_table_v` algebra for used rows; printed $\beta_1$ is reference-only |
| `phenomd.phase.pn.v1` | Eqs. (27), (46)–(53) | `pn_phase_coefficients`, `taylorf2_expr` | Torch/JAX TaylorF2 coefficients |
| `phenomd.phase.inspiral.v1` | Eq. (28) | `inspiral_expr` | Unwrapped phase in inspiral |
| `phenomd.phase.intermediate.v1` | Eqs. (15)–(16) | `intermediate_shape`, `phase_pieces` | Integrated derivative and unwrapped phase |
| `phenomd.phase.merger_ringdown.v1` | Eqs. (13)–(14) | `merger_ringdown_shape`, `phase_pieces` | Integrated derivative and unwrapped phase |
| `phenomd.phase.joins.v1` | Eq. (37) | `phase_pieces` | Symbolic $C^1$ joins and frontend values around both boundaries |
| `phenomd.phase.frontend_gauge.v1` | DiffGW convention | `peak_gauge_slope` | Coalescence and reference-frequency phase and frequency gradients |

`COMPONENT_BINDINGS` maps each ID to its source, reference function,
Torch/JAX entry points and maintained test names. The structural test resolves
those links. The numerical tests cover two selected equal/unequal-mass,
aligned-spin cases in FP64, using a declared supplied QNM pair rather than
assuming QNM fit provenance. They compare the unwrapped phase on both sides
of each join, the reference-frequency anchor, and selected frequency
gradients in each branch.
These results verify implementation agreement on the stated inputs. They do
not establish inherited fit provenance, QNM endpoint parity, complete
waveform or polarization agreement, or physical accuracy over the model domain.

Run `python -m pytest -q tests/test_imrphenomd_phase_math_reference.py`
for this slice. The maintained strict mathematical-reference CI gate also
collects it with both frontends required.
