# TEOB NQC derivative roundoff diagnostic

Using the saved `replay-native-output.npz` binary64 arrays (auxiliary spacing 0.1 and matching index 170), the degree-four derivative was recomputed from the polynomial moment equations. The centered row is (1, -8, 0, 8, -1)/(12 h); endpoint rows use the corresponding nearest-five-node degree-four polynomials. Repeated derivatives were applied as D(D), including D(D(p)).

The maximum current weighted-sum versus the independently assembled rational float64 stencil is 1.357e-10; the reconstructed coefficient difference is 4.525e-10. The maximum corrected original phase and amplitude differences are 4.650e-11 and 4.885e-15.

Using 80-digit arithmetic on the exact binary64 sampled values gives coefficient and corrected phase/amplitude deltas of 4.526e-10, 4.650e-11, and 4.274e-15. These are arithmetic roundoff scales on an unchanged sampled trajectory; they are far below 2.9e-4. This rules out roundoff in this solve on these inputs; it does not test target precision, input trajectory, timing, or reconstruction differences.
