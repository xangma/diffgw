# Pass 13: XHM nonmixing intermediate phase metadata

Date: 2026-09-13. This is a bounded source review of the nonmixing
`(2,1)`, `(3,3)`, and `(4,4)` intermediate phase fits. It uses only
García-Quirós et al., arXiv:2001.10914v2, its supplied mathematical fit
assets, the inherited native `imrphenom_xhm.py`/`xhm_fits.py`, and the two
earlier clean-room review notes. The native phase assembly now applies the
default f6 node correction described below; amplitude and 32 mixing branches
were not changed.

## Source result

Section V.B defines the intermediate phase-derivative ansatz (Eq. 5.4)

\[
 \frac{d\phi^{\rm Int}_{\ell m}}{df} =
 a_\lambda^{\ell m}\frac{f^{\ell m}_{\rm damp}}
 {(f^{\ell m}_{\rm damp})^2+(f-f^{\ell m}_{\rm ring})^2}
 +\sum_{k=0}^{4}\frac{a_k^{\ell m}}{f^k}.
\]

For the modes without significant mixing, the paper sets `a_3 = 0`, leaves
five coefficients, and imposes (Eq. 5.5)

\[
 \left.\frac{d\phi^{\rm Int}_{\ell m}}{df}\right|_{f_i^{\ell m}}
 = F_i^{\ell m},\qquad i\in[1,5].
\]

The text defines `F_i` as the physical phase derivative evaluated at each
collocation frequency through its parameter-space fit. Thus Eq. (5.5) gives
absolute physical targets; it does not, by itself, prove that ancillary files
named `p_i` were emitted without any preprocessing or relative conversion.
The supplied ancillary metadata does not document such a preprocessing step,
so no conversion is justified. The derivative and second-derivative boundary
conditions in Eq. (5.6) are the separate `(3,2)` mixing case and do not alter
Eq. (5.5). The native `p1`/`p2`/`p4` table mapping therefore remains inherited
and uncertified at the file-metadata level.

Table VII gives, with `b = beta(eta) f_Ins` and `e = f_End`,

```text
f1 = b
f2 = 1/4 ((sqrt(3)+2)b - (sqrt(3)-2)e)
f3 = 1/4 (e + 3b)
f4 = 1/2 (e + b)
f5 = 1/4 (3e + b)
f6 = 1/8 (7e + b)
```

For the nonmixing modes `f_End = f_ring` (the other branch in the table is
the `(3,2)` branch). Figure 8 labels the point at `f1` as the coincident
`C1`/`E1` endpoint, the second point `f2` as `C2`, and the remaining table
points as `E2=f3`, `E3=f4`, `E4=f5`, `E5=f6`. Its caption states that fits
are made at `[C1,C2,E2,E3,E4,E5]`; the default subset is
`[C1,C2,E2,E3,E5]`, and the sparse-calibration subset is
`[C1,C2,E3,E4,E5]`. Consequently:

```text
default five targets/nodes:  [F1@f1, F2@f2, F3@f3, F4@f4, F6@f6]
sparse five targets/nodes:   [F1@f1, F2@f2, F4@f4, F5@f5, F6@f6]
```

Before this correction, the native nonmixing assembly used `c_pts[:5]`, namely
`[f1,f2,f3,f4,f5]`, while solving against `phase_p1` through `phase_p5`.
That is neither Figure 8 subset. The actionable default correction, now
applied in the native phase assembly using the inherited `phase_p6_*` fits,
is to solve against
`[phase_p1, phase_p2, phase_p3, phase_p4, phase_p6]` at
`[c_pts[0], c_pts[1], c_pts[2], c_pts[3], c_pts[5]]`. This is a node/fit-index
correction only; no relative-target conversion is justified.
The focused `tests/test_pass13_xhm_phase.py` test captures the production
5x5 solve and reconstructs its nodes from the `1/f` basis column, verifies
that changing only `p6` changes normalized strain while `p5` is never called
for all three modes, and checks finite parameter gradients.

The paper gives no mathematical criterion identifying the parameter-space
regions with “fewer calibration waveforms” that select the sparse subset.
The supplied author archive includes the `(2,1)` files `3Dfit_p1_lm_21.m`
through `3Dfit_p5_lm_21.m`, but no p6 file and no metadata naming the
selection rule. The native fit table does expose `phase_p6_*`, but that
provenance is not established by the supplied ancillary files. Thus the
default f6 correction is source-consistent when those inherited p6 fits are
trusted; the sparse-subset switch and independent p6 ancillary provenance
remain explicitly unresolved.

## Primary-source hashes

```text
2001.10914v2.pdf                         00f3bdbe28a6ccde5cde3af82ac8fc95153ec6f03a4b071ce0ed57326e4e09ee
2001.10914v2.txt                         8041291797cf4facb95801762fbb80a59eaedf6211c354347d7e8a3392f17c86
anc/.../3Dfit_p1_lm_21.m                 428c1f23c4d1ca5e2c5935d4f0f0ce7c7e11cc45be79e64f4636064df5440935
anc/.../3Dfit_p2_lm_21.m                 a3ac2ef4b308abd7f2de64d05e45020c3e97f10160da9efd9c888c78b213bca0
anc/.../3Dfit_p3_lm_21.m                 60acb8aa7a6dcf1b25896d4fb2b2f51a03c19e3b86837868eb44fc30dd6f386e
anc/.../3Dfit_p4_lm_21.m                 9d350c7eae43ff62fce1193fb2d4e62c30c07b00f431ea6798ba40245b3063d7
anc/.../3Dfit_p5_lm_21.m                 b2c63176675569bed679bcd5c98e7cc91f825752985b070b047c26f1c79bd2f8
```

The governing references are arXiv:2001.10914v2 §V.B, Eqs. (5.4)--(5.6),
Table VII, and Fig. 8 (printed pp. 10 and 14 in the supplied PDF).
