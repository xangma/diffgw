# Pass 32: direct IMRPhenomPv2 opening-angle audit

Date: 2026-09-13

## Scope

This pass targeted the large strict IMRPhenomPv2 residual. The audit used only
the published precession equations in Pratten et al., arXiv:2004.06503v2 and
the independent TorchWave implementation. No LAL, PyCBC, TEOBResumS, or other
waveform implementation source was inspected or copied.

## Finding and correction

Pratten et al. define

\[
\chi_p = \frac{\max(A_1 S_{1\perp}, A_2 S_{2\perp})}{A_1 m_1^2}
\]

and then define the physical effective in-plane spin used by the opening angle
as `S_perp = m1^2 chi_p` (Eqs. 4.3--4.4). The direct `IMRPhenomP` path had
already included `A1` in `chi_p`, but multiplied by `A1` again when constructing
`S_perp`. This overstated the cone-driving spin by `A1`; for the representative
catalog mass ratio `q=35/25`, the factor is `A1=3.0714285714`.

The path now uses `S_perp = m1_norm^2 chi_p` and evaluates the signed angle as
`beta = atan2(S_perp, L + S_parallel)`. The latter is the published geometric
form and preserves the exact zero-cone limit. The in-plane norms also now retain
their exact zero rather than introducing a numerical floor.

## Validation

Added `test_imrphenomp_pv2_opening_angle_uses_physical_in_plane_spin`, which
checks the published identity and rejects the former extra-`A1` result. Focused
tests:

```text
pytest -q tests/test_imrphenomp.py tests/test_clean_room_precession_equations.py
24 passed in 3.03s
```

`git diff --check` passes. The remaining Pv2 parity gap may also include the
published precession-corrected final-spin/ringdown prescription; that is a
separate source-defined path and was not changed in this narrowly scoped pass.
