# IMRPhenomHM ringdown phase-shape component

## Scope and status

`hm.phase.ringdown.eq9-c1.v1` specifies the changed merger-ringdown arctangent shape in London's IMRPhenomHM phase, conditional on supplied PhenomD coefficients and QNM frequencies. It is a bounded executable research component, not a complete phase, mode, or waveform qualification. The printed formula and native D-carrier QNM adaptation have different claims, distinguished below.

## Inputs, conventions, and options

All frequencies are positive dimensionless $Mf$; $\eta>0$. Supply PhenomD coefficients $\alpha_4,\alpha_5$, HM QNM pairs $(f_{22}^{\rm RD,H},f_{22}^{\rm damp,H})$ and $(f_{\ell m}^{\rm RD,H},f_{\ell m}^{\rm damp,H})$, and the D carrier pair $(f_{22}^{\rm RD,D},f_{22}^{\rm damp,D})$. H and D identify QNM conventions, not different remnants. This component derives none of these inputs. The public HM assembly uses the correction only for its PhenomD carrier; its optional XAS carrier bypasses the branch.

## Mathematics and components

[London et al., arXiv:1708.00404v4, Eq. (14)](https://arxiv.org/pdf/1708.00404v4) modify the last arctangent of [Khan et al., arXiv:1508.07253v2, Eq. (14)](https://arxiv.org/html/1508.07253v2). With $R=f_{22}^{\rm RD,H}/f_{\ell m}^{\rm RD,H}$, the changed term is

$$
L_{\ell m}(f)=\frac{\alpha_4}{\eta}R\arctan\!\left(\frac{f-\alpha_5 f_{22}^{\rm RD,H}}{R f_{\ell m}^{\rm damp,H}}\right),
$$

and the original D term is

$$
D(f)=\frac{\alpha_4}{\eta}\arctan\!\left(\frac{f-\alpha_5 f_{22}^{\rm RD,D}}{f_{22}^{\rm damp,D}}\right).
$$

These form the literal London-minus-Khan delta only when both 22 QNM pairs coincide. The live Torch/JAX code permits distinct pairs. It uses HM ratios $R$ and $T=f_{\ell m}^{\rm damp,H}/f_{22}^{\rm damp,H}$, but anchors the absolute center and width to the D carrier:

$$
W=R T f_{22}^{\rm damp,D},\quad C=\alpha_5 f_{22}^{\rm RD,D},\quad
\delta(f)=\frac{\alpha_4}{\eta}\left[R\arctan\!\left(\frac{f-C}{W}\right)-\arctan\!\left(\frac{f-C}{f_{22}^{\rm damp,D}}\right)\right].
$$

This is a native carrier-consistency adaptation, not another equation printed by London. It reduces to $L_{\ell m}-D$ when the 22 pairs coincide. PhenomD begins its MR phase at $J=f_{22}^{\rm RD,D}/2$ (Khan et al., Sec. V.A). The native correction removes the value and tangent there:

$$
\Delta(f)=\begin{cases}0,&f\leq J,\\
\delta(f)-\delta(J)-(f-J)\delta'(J),&f>J.
\end{cases}
$$

Thus both value and slope join at zero. This affine attachment is a native choice, not another HM fit. For the 22 control, $R=T=1$ and $\Delta\equiv0$. Other Eq. (14) terms remain inherited from D.

| Component ID/version | Source | Variant | Status | Scalar reference | Torch | JAX | Maintained check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hm.phase.ringdown.eq9-c1.v1` | London Eq. (14); Khan Eq. (14), Sec. V.A | supplied alpha/QNM, D-carrier C1 | executable, conditional | `carrier_delta_c1` | `_production_eq14_delta_c1` | `_production_eq14_delta_c1` | `test_imrphenomhm_ringdown_phase_math_reference.py` |

## Implementation mapping

The independent 70-digit mpmath evaluator is `src/diffgw_reference/imrphenomhm_ringdown_phase.py`; it imports no frontend, fit, or QNM code. The maintained test calls the live Torch and JAX `_production_eq14_delta_c1` helpers, each called by public HM D-carrier assembly. All parameters are supplied, so passing cannot establish their physical or fit provenance.

## Evidence and limitations

The [len validation receipt](../validation/imrphenomhm_ringdown_phase_20260927.md) records passing checks of the literal paper limit, sampled values on both sides of the join and through the arctangent center, frequency and $\alpha_4$ gradients, C1 value/slope, and a 22-mode control. Unequal HM/D 22 QNM fixtures expose the native adaptation rather than silently identifying conventions. The checks do not validate PhenomD $\alpha$ surfaces, remnant/QNM calculations, mode-dependent QNM choice, full phase gauge, mapped-phase attachment, complete modes, or NR/LAL accuracy.

## Reproduce the checks

With reference, Torch, JAX, and test extras installed, run `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q tests/test_imrphenomhm_ringdown_phase_math_reference.py --reference-device=cpu -ra`. Strict mathematical-reference CI discovers this file. A changed equation or convention requires a versioned source/fixture review.
