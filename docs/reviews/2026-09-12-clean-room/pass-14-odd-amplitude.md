# Pass 14: odd-m amplitude targets

The author excerpt from arXiv:2001.10914v2, Sec. V.A.2, states that odd-
`m` calibration surfaces may carry negative values so that the fit remains
smooth, and that reconstructed amplitudes take the absolute value. Its Fig. 4
example is the first intermediate `(3,3)` collocation node.

The native consumer previously passed raw `amp_insp1/2/3` values into the
inspiral collocation reconstruction and converted negative `amp_int1/2`
values with `clamp(..., min=1e-12)`. Evaluating the author expressions at the
chosen equal-mass, unequal-spin point `chi1=0.6`, `chi2=-0.3`, the raw `(2,1)` intermediate fits
are negative. The consumer now folds odd-`m` inspiral and intermediate
collocation targets with `abs` before reconstruction. Ringdown already used
`abs(alambda_fit)`, and raw `XHMParameterFits` expressions remain unchanged.

Focused tests capture the production inspiral target and six-condition
intermediate solve target, check finite nonzero gradients away from zero, and
verify that even-`m` target handling retains its prior sign behavior.
