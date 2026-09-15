# Pass 13: XHM intermediate amplitudes

## Source determination

The García-Quirós et al. manuscript (arXiv:2001.10914v2) defines
`A0_22(f) = pi * sqrt(2 eta / 3) * (pi f)^(-7/6)` in Eq. (4.5), and
defines the inspiral amplitude ratio `H_lm = |A^SPA_lm| / A0_22(f)` in
Eq. (4.6). The intermediate region starts at Eq. (5.1). Its fit in Eq. (5.3)
is instead

`A^Inter_lm / A0 = 1 / (delta0 + delta1 f + ... + delta5 f^5)`,

with `A0 = pi * sqrt(2 eta / 3)`, independent of frequency. Tables III and IV
therefore give the dimensionless `A^Inter_lm / A0` node targets. If the
production polynomial stores inverse node values `p_i`, its conditions are

`p_i = 1 / (A^Inter_lm(f_i) / A0)` and
`H_lm(f) = (pi f)^(7/6) * A^Inter_lm(f) / A0`.

The previous implementation multiplied the inverse node values by
`(pi f_i)^(7/6)`, treating the Table III/IV targets as `H_lm`. Since those
targets are actually `A^Inter/A0`, that node construction made the evaluated
ratio equal to `A^Inter/A0` and omitted the required `(pi f)^(7/6)` conversion.
The intermediate nodes now use `p_i = 1 / (A^Inter / A0)`; the existing
evaluation supplies the single `(pi f)^(7/6)` factor.

The ringdown fit is already expressed directly relative to `A0_22(f)` in
Eq. (6.2), so no ringdown normalization change is needed.

Table VI and Sec. V.A.3 distinguish two fallbacks. A parameter-space veto
removes both inner values and keeps the endpoint derivatives, giving an
endpoint Hermite cubic. A denominator crossing after the veto checks removes
the endpoint derivatives while retaining all four positive values, giving a
value cubic through both inner nodes. If that cubic still fails, the
reconstruction drops to the endpoint linear fit. The implementation selects
each reconstruction for the complete intrinsic interval. Its existing pole
detector samples 11 equally spaced points and flags values below 5% of the
smaller endpoint denominator; this remains a bounded heuristic rather than an
exact root search.

The native `(3,2)` join contract test uses the published default parameters and
independently captures the production six-condition solve. It verifies that the
parameter veto is inactive while the 11-point/5% detector is active, so the
endpoint derivatives are intentionally not required at the two amplitude joins;
C0 and all phase-join checks remain required. A separate production test sets
the two inner fit values from the actual endpoint Hermite polynomial, retaining
physical boundary values and derivatives. It independently captures the new
six-condition solve and proves its denominator stays above the detector
threshold. The complete waveform then satisfies C0 and C1 amplitude conditions
at both joins. Parent review replaced a polynomial-only control with this
waveform-level check; the control makes no claim about calibration accuracy.

## Implementation and checks

`src/torchwave/approximants/imrphenom_xhm.py` labels the intermediate fit
values as `v_int1_fit` and `v_int2_fit` and converts them to inverse polynomial
nodes with `1 / v_int*`. The focused production tests independently reconstruct
both (2,1) nodes, their physical `A0_22(f)`, and the published `A^Inter/A0` fit
targets. The normalized unprojected strains match with zero absolute tolerance
(`rtol=5e-12`, `atol=0`), and frequency, mass, and spin gradients are finite and
nonzero.

Validation command:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  -m pytest -q tests/test_pass13_xhm_amplitude.py
```

Result: `4 passed`.

Parent final focused integration, including `test_clean_higher_modes.py`,
`test_pass12_higher_modes.py`, and both pass-13 suites: **57 passed in 5.17 s**.
