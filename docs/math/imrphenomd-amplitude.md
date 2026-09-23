# IMRPhenomD amplitude and joins, first reference slice

`imrphenomd-amplitude-joins-paper-precision/v1` is an executable specification
in [`src/diffgw_reference/imrphenomd_amplitude.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomd_amplitude.py).
Its primary source is [Khan et al., Paper II, arXiv:1508.07253v2](https://arxiv.org/html/1508.07253v2).
This slice specifies the dominant-mode **scaled amplitude** and seven printed
calibration rows. It takes the total-mass-unit ringdown frequency $f_{\rm RD}$
and damping frequency $f_{\rm damp}$ as inputs. Remnant fits, Kerr QNMs, phase,
polarizations, the $Mf=0.2$ implementation cutoff and optional tapers are
outside this version's mathematical claim.
The [remnant reference slice](imrphenomd-remnant-qnm.md) specifies the remnant
fits and conditional QNM inputs used at this boundary. The
[phase reference slice](imrphenomd-phase.md) specifies the intrinsic phase
branches and their joins.

The dimensionless frequency is $u=Mf$, with $M$ in seconds when $f$ is in Hz.
The amplitude below is $A/A_0$; the common Newtonian factor $A_0\propto
f^{-7/6}$ is restored by the frontends. For masses $m_1\geq m_2>0$,
$\eta=m_1m_2/M^2$, $\delta=(m_1-m_2)/M$,
$\chi_s=(\chi_1+\chi_2)/2$, and $\chi_a=(\chi_1-\chi_2)/2$.
Paper II Eqs. (2)–(4) give

$$\chi_{\rm PN}=\frac{m_1\chi_1+m_2\chi_2}{M}
-\frac{38\eta}{113}(\chi_1+\chi_2).$$

Table 5 uses this **unscaled** $\chi_{\rm PN}$, not the normalized
$\hat\chi$ discussed elsewhere in the paper. Its seven amplitude rows
$\rho_{1,2,3},v_2,\gamma_{1,2,3}$ are transcribed as exact decimal strings,
using only the digits printed in Table 5. Each row $c_0,\ldots,c_{10}$ enters
Eq. (31), with $x=\chi_{\rm PN}-1$:

$$\Lambda=c_0+c_1\eta+x(c_2+c_3\eta+c_4\eta^2)
+x^2(c_5+c_6\eta+c_7\eta^2)+x^3(c_8+c_9\eta+c_{10}\eta^2).$$

Eqs. (29)–(30) and Appendix B Eqs. (54)–(60) give the inspiral branch

$$a_{\rm ins}(u)=\sum_{k=0}^6\mathcal A_k(\pi u)^{k/3}
+\rho_1u^{7/3}+\rho_2u^{8/3}+\rho_3u^3.$$

The seven $\mathcal A_k$ are written as exact rational and symbolic $\pi$
terms in the reference module. Eq. (19) gives

$$a_{\rm MR}(u)=\frac{\gamma_1\gamma_3f_{\rm damp}}
{(u-f_{\rm RD})^2+(\gamma_3f_{\rm damp})^2}
\exp\left[-\frac{\gamma_2(u-f_{\rm RD})}{\gamma_3f_{\rm damp}}\right].$$

The lower join is $u_1=0.014$; the upper join $u_3$ is the peak in Eq. (20)
where $|\gamma_2|<1$. The existing Torch/JAX implementations continue that
formula at $|\gamma_2|\geq1$ by setting the square root to zero. That gives a
real join frequency but **not** a stationary point of Eq. (19); the reference
names this separately as `peak_compatibility` and uses the true derivative
$a'_{\rm MR}(u_3)$ in the join. This continuation is an implementation choice,
not a result established by Eq. (20).

For $t=(u-u_1)/(u_3-u_1)$ and $h=u_3-u_1$, the middle branch is the unique
quartic matching $a_{\rm ins}$ and its derivative at $u_1$,
$a_{\rm MR}$ and its derivative at $u_3$, and fitted $v_2$ at
$u_2=(u_1+u_3)/2$ (Eqs. (21)–(26), Table 2). The reference constructs it as
a cubic Hermite interpolant $H(t)$ plus
$16[v_2-H(1/2)]t^2(1-t)^2$. This differs algebraically from the frontend
Vandermonde solve but defines the same quartic. Both branches and their first
derivatives therefore meet at each join.

## Evidence and checks

| Component IDs | Paper anchor | Comparison |
| --- | --- | --- |
| `phenomd.amp.fit.*.v1` | Eq. (31), Table 5 | Printed-row transcription and hierarchical evaluation on both frontends |
| `phenomd.amp.spin_coordinates.v1` | Eqs. (2)–(4) | Exact spin coordinate construction in test cases |
| `phenomd.amp.pn.v1`, `phenomd.amp.inspiral.v1` | Eqs. (29)–(30), (54)–(60) | All PN coefficients and inspiral waveform-amplitude ratios |
| `phenomd.amp.merger_ringdown.v1`, `phenomd.amp.peak.v1` | Eqs. (19)–(20) | Peak formula and merger-ringdown amplitude ratios; continuation called out separately |
| `phenomd.amp.intermediate.v1` | Eqs. (21)–(26), Table 2 | Exact five-condition identity, interior values and endpoint slopes against Torch/JAX |

The backend bindings are [`torchwave/approximants/imrphenomd.py`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenomd.py)
and [`jaxwave/approximants/imrphenomd.py`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenomd.py):
`eval_table_v` evaluates the fit IDs, `pn_amplitude_coefficients` the PN ID,
`amplitude_peak_frequency` the peak ID, and
`intermediate_amplitude_coeffs` with `compute_slope_peak` the join ID.
`forward` (Torch) and `evaluate` (JAX) assemble the spin coordinates and
three amplitude branches. The maintained
[`test_imrphenomd_math_reference.py`](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomd_math_reference.py)
checks every mapped component, including values on both sides of each join
and frequency derivatives in all three open regions.

The production [`TABLE_V_UNROUNDED_COEFFS`](https://github.com/xangma/diffgw/blob/main/src/diffgw_core/fits.py)
stores additional inherited digits whose publication provenance remains
unresolved. Tests pass that table **explicitly** to the independent Eq. (31)
formula to verify implementation algebra, while the reference's own default
rows contain only published digits. Agreement under the supplied production
rows is not evidence that the extra digits appear in the paper. The complete
amplitude test compares ratios across all three regions, canceling the shared
SI conversion and orientation factors. It qualifies the amplitude equations
conditional on the supplied remnant frequencies and calibration data; it does
not qualify the complete waveform or its physical accuracy.

Run `pytest -q tests/test_imrphenomd_math_reference.py`. The maintained strict
`--require-math-reference` gate also collects this module automatically.
