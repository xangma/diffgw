# Predeclared NQC second-derivative experiment

This declaration precedes the five-level convergence runs. The evidence used
to select the candidate is `nqc_second_derivatives_two_rows.json` (SHA-256
`32cdd77e4d337c568b71adf1ea01e82549f2acaf2d6f86cf9b215b8e0c4c903f`).
It compares two exact-hash frozen rows with DOP853 dense output of the **same
provisional equations**. It does not establish a historical SEOBNRv4 rule.

For each fine-grid match time, select samples with `|t-t_match| <= 0.5 M`,
requiring at least nine points on both sides collectively and at least one on
each side. Fit a degree-six polynomial in `(t-t_match)/0.5 M`, separately to
`A`, `A q_1`, `A q_2`, `A q_3`, and the unwrapped waveform phase. Subtract a
constant before fitting for conditioning. Replace **only** their second
derivatives in the NQC amplitude matching matrix/right-hand side and phase
right-hand side. Keep the existing local phase-basis matching matrix, spline
values and first derivatives, NQC targets, all event rules, and other waveform
operations unchanged. No coefficient fitting or post-run window/order tuning.

Run the unchanged `qualify_generator.py` five-level numerical gates on the
frozen nonspinning `(30,20,0,0)` and aligned-spin campaign rows, with Hermite
resampling and a fixed `165 M` restart. Tolerances are
`(atol,rtol)=(1e-12,1e-11),(1e-13,1e-12),(1e-14,1e-13)` at `dt=0.5 M`,
followed by `dt=0.25,0.125 M` at the tightest tolerances. For each ODE and
grid refined pair, require complex and amplitude relative L2 at most `1e-6`,
complex error nonincreasing from the preceding pair (allowing `1e-12`), and
peak, match, and boundary-time differences at most `0.05 M`. Report each gate
even if it fails. A pass is numerical evidence only; the physics and generator
remain unqualified, and no ROM training or release promotion follows.
