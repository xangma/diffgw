# Pass 18: NSBH remnant-frequency classification

## Primary derivation

The primary source is Thompson et al., arXiv:2002.08383v2, “Modeling the
gravitational-wave signature of neutron-star–black-hole mergers”
([arXiv abstract](https://arxiv.org/abs/2002.08383v2)).  The official source
archive used for this audit is
`https://arxiv.org/e-print/2002.08383v2`, SHA-256
`bf81b62a2397ca291e4847e979c62f83f2435e29f470c12baa111041f7377855`.
The extracted `paper.tex` is the sole primary artifact used here.

Appendix A gives the ordered construction.  After the disruption frequency
and torus mass, step 4 computes both remnant quantities from the NSBH fit
(paper Eq. (A7), reproduced as `eq:remnant-model`):

\[
 F(\eta,\chi,\Lambda)=F_{\rm BBH}(\eta,\chi)
 \frac{1+p_1\Lambda+p_2\Lambda^2}
 {(1+p_3^2\Lambda)^2}.
\]

The fit is applied to final mass and final spin.  Step 5 then obtains the
ringdown frequency from that remnant,

\[
 f_{\rm RD}=\frac{\Re\tilde\omega(\chi_f)}{2\pi M_f},
\]

using the stated Kerr QNM fit (paper Eqs. (A8)–(A10)).  Step 6 classifies
with the table condition `f_tide >= f_RD` (a closed boundary), together with
`M_b,torus = 0` versus `> 0`.  The prose immediately below the table repeats
the same conditions.  Therefore the frequency in the classification test is
the NSBH-remnant frequency, including its Lambda dependence.  The BBH
quantity is only the baseline inside the NSBH fit and is recovered at
Lambda=0; it is not a separate non-disruptive classification remnant.

## Native audit and change

The native amplitude flow previously computed both the BBH and Zappa remnant
quantities, compared `f_tide` against the BBH ringdown frequency, and selected
the BBH remnant for the `f_tide >= f_RD` and no-torus branch.  At finite Lambda
this makes the classification boundary differ from Appendix A.  For example,
`(m1,m2,chi1,Lambda)=(5.6,1.4,0,800)` has `f_tide` between the NSBH and BBH
ringdown frequencies in float64, so the two branches select different table
columns.

The runtime now computes `(M_f, chi_f)` from `zappa_remnant_properties` for
all classes, computes one Kerr ringdown frequency from that remnant, and uses
the source's closed `f_tide >= f_RD` comparison.  Lambda=0 still reproduces
the BBH limit through the fit.  The standalone `nsbh_frequencies` helper was
already using the NSBH remnant and remains consistent with the runtime.

## Tests and limits

`tests/test_pass18_nsbh_classification.py` checks the source-derived boundary
on a separating configuration, batched frequency evaluation, and finite
waveform gradients.  Focused validation:

```
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_pass18_nsbh_classification.py
```

The model contains fitted remnant and QNM formulae, so this establishes
source-consistent construction and tensor behavior rather than an exact
algebraic or numerical certification of merger boundaries.  No parity output,
residual, external implementation, or public binary was used.
