# TEOB post-PA ODE convergence

The same PA handoff and physical inputs were replayed with predeclared
`(atol, rtol)` values `(1e-13, 1e-11)`, `(1e-14, 1e-12)`, and `(1e-15, 1e-13)`.
Fixed-step continuation, NQC targets, event rules, and ringdown assembly were
unchanged.  The native accepted grids had 668, 1012, and 1559 samples.

The cache-only review in [reviewed-report.json](reviewed-report.json) compares
adaptive states by natural-cubic interpolation on their shared public grid;
it compares modes only on the exact common public subset.  Directly from
tight_0p1 to tight_0p01, pre-NQC mode changes by relative L2 `2.68e-8`, while
the processed NQC mode changes by `2.16e-3`; state-component maxima are
`[1.89e-7, 2.45e-7, 2.43e-8, 1.50e-7]`.  The attachment time shifts by
`-0.0274 M`. Native states and pre-NQC modes differ little at these sampled
tolerances, whereas processing amplifies the differences. This does not
establish asymptotic convergence or exclude amplification of integration errors
through derivatives or interpolation. NQC/event/model attribution remains
unresolved; indices on differing adaptive grids are not comparable events.

Against the public full mode, relative L2 is `1.7625e-3`, `4.0396e-3`, and
`2.0543e-3` in tolerance order: tighter tolerances do not consistently improve
agreement. This is a 30+20 nonspinning, approximately 14.45 Hz diagnostic,
distinct from the earlier 20 Hz frequency-domain comparison.

The clean NQC API derives its auxiliary grid from each input trajectory and
derives `t_nqc` from that grid's frequency peak.  It has no argument to inject
a baseline event time or auxiliary grid.  Freezing those independently would
require a new assembly implementation, so no extra control was run.  The
original generating `report.json` and its source hash are preserved. Its
generic approach-to-baseline convergence criterion is superseded by this
successive-level review: convergence need not approach the coarsest result.
No runtime defaults were changed.
