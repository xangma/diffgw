# Pass 32: independent IMRPhenomPv2 review

Date: 2026-09-13

This review used the published precession equations in Pratten et al.,
arXiv:2004.06503v2, the TorchWave implementation, and TorchWave tests only.
No LAL, PyCBC, TEOBResumS, or other waveform implementation source was
inspected or copied.

The opening-angle correction is source-supported. Equations (4.3)--(4.4)
define `chi_p` with the `A1` weighting and then use the physical
`S_perp = m1^2 chi_p` in `beta = atan2(S_perp, L + S_parallel)`. The patch's
removal of the duplicate `A1`, use of `atan2`, and exact zero-cone value are
therefore correct. The aligned-spin limit remains the zero-angle Wigner
projection.

The review found one differentiability defect in the exact-zero implementation.
`sqrt(x*x+y*y)` and the fallback `atan2(0, 0)` can produce NaN gradients even
when the waveform itself is finite. This occurred for aligned inputs at
positive frequencies and affected masses as well as in-plane spins through the
precession graph. The implementation now selects a finite square-root branch
at the origin and uses a finite `(0, 1)` azimuth convention when no in-plane
direction exists. Both preserve the exact zero magnitude and zero-cone output;
they add no physical floor or fitted coefficient.

Validation:

```text
77 passed
git diff --check passed
```

The focused run covered `test_imrphenomp.py`,
`test_clean_room_precession_equations.py`, and
`test_clean_room_precession_conventions.py`. This review does not establish
full Pv2 parity; the prescription-specific frame and final-spin/ringdown
paths remain separate questions.
