# Pass 9: XP/XPHM precession dispatch and remnant boundary

## Scope

This pass checked the native XP/XPHM precession dispatch and the parameter
boundary between the precession angles and the aligned-spin carrier. The
review used the native source, the earlier clean-room reports, and the
primary prescriptions in Pratten et al., arXiv:2004.06503v2, Sec. IV D and
Table V, and Chatziioannou et al., arXiv:1703.03967v2, Appendix A. No
LAL/PyCBC source, waveform oracle, or residual fit was used.

## Dispatch repair

`evaluate_msa_precession` declares `prec_version=223`. Before this pass, an
explicit `prec_version=None` was resolved with the NNLO method name, selecting
102. Thus an omitted argument and an explicit `None` produced different MSA
dispatches and different angles. The evaluator now resolves `None` with the
`"msa"` method, so it selects 223 consistently with its declared default.
Explicit version validation and the public XP/XPHM wrapper dispatch remain
unchanged.

The implementation change is in
`src/torchwave/approximants/precession_msa.py`. The regression is in
`tests/test_clean_room_precession_conventions.py`:
`test_msa_evaluator_explicit_none_matches_declared_default` compares the
omitted and explicit-`None` calls, including the returned carrier phase.

Validation used the required environment and the focused 79-test set:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
tests/test_clean_room_precession_conventions.py \
tests/test_clean_room_msa.py \
tests/test_phenomxp_msa_reconstruction.py
```

Result: **79 passed**.

## Published remnant prescription: feasibility boundary

The primary IMRPhenomXPHM paper does specify the remnant *models*: its Eq.
(4.25)--(4.26) uses the effective `chi_p` prescription (Table V,
`FinalSpinMod=0`) for NNLO, while Eqs. (4.20) and (4.29) use the
precession-averaged couplings and the corresponding spin norm (Table V,
`FinalSpinMod=3`) for MSA. Those equations are sufficient to define the
mathematical formulas once their evaluation state is supplied.

They do not, however, fully specify a drop-in change for this native carrier.
The current MSA evaluator returns angles, not the averaged couplings needed
by Eq. (4.20), and it has no separate merger/attachment state argument. The
current XAS/XHM carriers compute the aligned `a_parallel_fin` and `M_fin`
internally and expose no final-spin override. The paper also gives no
`FinalSpinMod` mapping for this branch's ODE method or its fallback rows.

The exact unresolved choice is therefore whether the MSA averages should be
sampled at the current reference state (`v0`) or at a separately specified
merger/attachment state, plus what remnant policy ODE/fallback dispatch should
use. Choosing either without a native contract or a primary-paper statement
would invent behavior. No remnant code was changed in this pass.

If that contract is later supplied, the paper-supported implementation plan
is to expose the existing MSA averaged couplings, implement the NNLO/Table-V
and MSA/Table-V formulas using the existing aligned `M_fin` and
`a_parallel_fin` fits, pass an optional final-spin value into both carriers,
and dispatch only the explicitly specified method rows. No inherited fit
tables or thresholds need to change.
