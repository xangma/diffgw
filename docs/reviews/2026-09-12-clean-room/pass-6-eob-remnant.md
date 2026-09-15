# Published EOB remnant inputs

Date: 2026-09-12. This pass adds only `eob/remnant.py`, its focused tests,
and this report. No waveform driver, model default, or existing remnant fit
is changed. The implementation uses primary mathematical publications;
no external implementation, binary oracle, NR-result fitting, or old
calibration specification was used.

## APIs and scope

`eob_final_mass_fraction(mass1, mass2, chi1, chi2)` returns
`M_final/(mass1+mass2)`. `hofmann_final_spin(mass1, mass2, chi1, chi2,
fit_order=...)` returns the **signed** final-spin projection along orbital
angular momentum. The keyword `fit_order` is mandatory and accepts exactly
`(1,2)`, `(3,3)`, or `(3,4)`. These are the three published alternatives;
no SEOBNRv4 selection is inferred.

Masses must be positive and finite, in any common unit. Spins are finite
signed projections in `[-1,1]`. Inputs broadcast as parameter batches.
Float32/float64 tensor types are promoted on their common device; Python
scalars follow that tensor type, or use float64 when all inputs are Python
scalars. The functions preserve ordinary autograd and work under
`no_grad` and `inference_mode`. Mixed tensor devices and unsupported tensor
types fail explicitly. Mass fractions are computed after scaling both
masses by their maximum, avoiding overflow from squaring dimensional
masses. The formulas are symmetric under simultaneous mass/spin exchange.

These APIs implement scalar projections. Hofmann's generic precessing
spin magnitude/direction extension and the choice of an evolution time at
which spin directions are evaluated are outside this component.

## Final mass: Ossokine Eq. (3.5)

The directly inspected author-institute PDF prints the expression in
Sec. III C, journal page 044055-9, with the definition of `a` on the next
page. It identifies this as the mass fit used in preceding EOB models.
With `Xi=mi/(m1+m2)` and `nu=X1*X2`, it reduces algebraically to

```
a = X1^2*chi1 + X2^2*chi2
chi = a/(1-2*nu)
E_equal = 0.00258 - 0.0773/(chi-1.6939)
M_final/M = 1 - nu*(1-4*nu)*(1-E_ISCO(a)) - 16*nu^2*E_equal.
```

The implementation evaluates `chi` equivalently as
`(m1^2*chi1+m2^2*chi2)/(m1^2+m2^2)` after mass scaling. Thus its denominator
is positive, and the pole at 1.6939 lies outside the physical spin domain.
`E_ISCO` is the **total specific energy** `sqrt(1-2/(3*r_ISCO))`, consistent
with Hofmann Eq. (2) and the test-particle radiated-energy limit. The prose
following Ossokine's equation calls it binding energy; inserting `1-E`
as the symbol `E_ISCO` would give the wrong test-particle limit. At equal
mass, the ISCO term vanishes and the radiation is exactly `E_equal`.
The three fitted constants retain the printed precision.

## Final spin: Hofmann Eq. (7), Table 1, Eq. (11)

The directly inspected PDF pages 2–3 provide the entire aligned formula:

```
a_eff = a + xi*nu*(chi1+chi2)
a_final = a + nu*[L_ISCO(a_eff)-2*a*(E_ISCO(a_eff)-1)]
          + sum_(i=0..nM,j=0..nJ) k_ij*nu^(2+i)*a_eff^j.
```

All Table 1 entries were transcribed from its three blocks, including signs
and their printed decimal precision. `xi` is respectively `0.41616`,
`0.463926`, and `0.474046`. The absent `k00` is calculated from Eq. (11),
using its central equal-mass nonspin value `0.68646`:

```
k00 = 16*[0.68646-sqrt(3)/2-sum_(i=1..nM) k_i0/4^(2+i)].
```

The resulting double-precision values are approximately
`-3.8211589605510183`, `-5.831643335551019`, and
`-5.977230835551018`, in block order. The paper's `±0.00004` is an
uncertainty on the constraint, not another fitted implementation constant.
No unprinted coefficient digits are inferred. Nested Horner evaluation
computes the same two-variable polynomial. The signed result is returned
without an absolute value or clipping.

Which Table 1 block and coefficient precision the exact SEOBNRv4 target
selected remains unresolved by the inspected primary papers. Requiring
`fit_order` keeps that choice explicit. This component alone establishes
neither a complete SEOBNRv4 waveform nor implementation parity.

## Signed Kerr ISCO and derivatives

The shared private helper implements Hofmann Eqs. (2)–(6), with positive
orbital angular momentum and signed Kerr spin `a`. It returns radius,
total specific energy, and positive specific orbital angular momentum.
The endpoint/zero radii are exactly `r(-1)=9`, `r(0)=6`, and `r(1)=1`.

Directly evaluating `sign(a)*sqrt(3-Z1)` loses precision and has a spurious
autograd singularity near zero. An algebraic rationalization removes it.
Let `u=cbrt(1+a)`, `v=cbrt(1-a)`, `t=u*v`, `s=u+v`, and
`d=u^2+u*v+v^2`. Then

```
(3-Z1)/a^2 = [s+4/(2+s)]/(1+t+t^2) - 4/[(2+s)*d^2].
r = 3+Z2-a*sqrt(((3-Z1)/a^2)*(3+Z1+2*Z2)).
```

The right-hand ratio is regular at zero, with value `8/9`; the identities
`1-t=a^2/(1+t+t^2)` and `u-v=2*a/d` derive this expression directly.
No fitted interpolation or arbitrary spin threshold is introduced.

The custom backward differentiates the defining marginal-stability
equation `r^2-6*r+8*a*sqrt(r)-3*a^2=0`:

```
dr/da = (6*a-8*sqrt(r))/(2*r-6+4*a/sqrt(r)).
```

It gives the finite physical derivatives `-45/16` at `a=-1` and
`-4*sqrt(2/3)` at zero. At exactly `a=+1`, the ISCO derivative diverges;
the helper preserves that mathematical limit. There is no promise of
finite gradients when an effective ISCO spin numerically reaches this
endpoint. Ordinary nonzero binary mass ratios keep the effective spins
inside the endpoints, including extremal component spins, subject to
floating-point resolution at extreme mass ratios.

## Verification

Focused run:

```
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest tests/test_clean_room_eob_remnant.py -q
```

Result: **59 passed, 4 skipped**. All four CUDA checks were skipped because
CUDA is unavailable; all four corresponding float32 MPS checks passed.
CPU checks include both float32 and float64, type promotion, and normal,
no-grad, and inference execution.

Numerical checks compare signed ISCO values to independent 75-digit
bisection roots of the marginal-stability equation, including zero,
`±1`, and values within `1e-12` of those points. They verify physical
derivatives, first and second autograd derivatives, the equal-mass mass
and spin constraints, first-order test-particle limits for all three spin
blocks, batch/scalar agreement, exchange symmetry, mass-unit factors
from `1e-250` to `1e250`, signed negative remnants, all intrinsic
gradients, and explicit invalid-input failures. These establish equation
implementation properties, not NR accuracy or target-model parity.

## Primary-source provenance

Fresh primary PDFs retrieved on 2026-09-12 for the preceding source
assessment were reopened and visually inspected in this pass. Relevant
pages: Ossokine Eq. (3.5) on 044055-9 and its following definition;
Hofmann pp. 2–3, Eqs. (2)–(7), Table 1 and Eq. (11).

| Source | PDF SHA-256 |
|---|---|
| [Ossokine et al., PRD 102, 044055 (2020), author-institute copy](https://pure.mpg.de/rest/items/item_3230468_5/component/file_3252882/content) | `a8dcecbc5269e68f8a121a431bc6759123ba4bee179caf15a94ddc07ee70f7b3` |
| [Hofmann, Barausse & Rezzolla, arXiv:1605.01938v2](https://arxiv.org/pdf/1605.01938v2) | `16bceff025a3084f5f424bd992b2659f92419da119349670ac0010d8ae80c8a3` |
