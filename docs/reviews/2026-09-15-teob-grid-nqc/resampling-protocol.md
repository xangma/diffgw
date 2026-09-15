# Follow-up: pre-attachment output interpolation

Declared after the derivative experiment and corrected public-TD localization,
before this control runs. The catalog's late/pre-NQC segment has one output
sample and contains 98.35% of its joint TD squared residual. This motivates a
separate output-interpolation control; it does not prove a cause.

Rebuild the nominal 0.1 M sampled NQC and ringdown baseline from the same four
accepted caches. Require exact replay of each nominal TD/FD cache. Keep its
coefficients, grid peak, attachment, output times, support, and ringdown mode.
Before the actual grid attachment time only, replace output amplitude, phase,
or both with natural-cubic interpolation of the corrected accepted series.
This is the accepted-series interpolation already used by the opt-in
continuous ringdown path, separated from its changed event and attachment.
Retain baseline samples at and after attachment exactly. Use the same strict
FD harness; retain all outcomes and do not fit alignment or coefficients.

No runtime promotion is implied by a better isolated score. This is a bounded
test of the join/resampling contribution with fixed physical inputs.
