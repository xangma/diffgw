# TaylorF2 eccentric phase at order \(e_0^2\)

## Scope and status

Component specification **`ecc.phase-3pn-e2/v1`** covers the secular,
dominant-harmonic stationary-phase correction through relative 3PN and order
\(e_0^2\). A separate **`ecc.phase-fref-gauge/v1`** specifies the native phase
origin. The scalar [SymPy reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_eccentric.py)
transcribes [Moore et al. (2016), Eq. (6.26)](https://arxiv.org/pdf/1605.00304v2)
without importing either frontend or LAL. This is a component-level result,
not qualification of the complete waveform or the optional amplitude factor.

## Inputs, conventions, and options

Input masses are in solar masses and both frequencies are positive GW
frequencies in Hz. Let \(M=m_1+m_2\), \(\eta=m_1m_2/M^2\),
\(v=(\pi GMf/c^3)^{1/3}\), and \(v_0=v(f_{0,\rm ecc})\). The eccentricity
\(e_0\) is defined at \(f_{0,\rm ecc}\); it is not defined at the separate
waveform phase origin \(f_{\rm ref}\). The paper's leading-eccentricity
approximation was estimated to apply to comparable-mass binaries at roughly
\(e_0\lesssim0.1\), with a more restrictive extreme-mass-ratio range. The
native API accepts a wider input range; that acceptance is not a validity
claim. The focused gate uses CPU float64, positive frequencies, and no ISCO
cutoff. No numerical table or trained asset enters this component.

The native `ecc_order` values 0, 2, 3, 4, 5, and 6 retain the Newtonian,
1PN, 1.5PN, 2PN, 2.5PN, and 3PN eccentric terms respectively. Order 6
means relative **3PN**, not \(e_0^6\). An explicit `f0_ecc` takes precedence
over `f_lower`, then a positive `f_ref`, then the lowest positive grid value,
with a 20 Hz fallback on an empty-positive grid. The gate supplies `f0_ecc`
explicitly so this grid-dependent default is outside the paper check.

## Mathematics and components

Eq. (6.26) gives the eccentric addition to the SPA phase:

\[
\Delta\Psi_{\rm paper}(f;f_0) =
-\frac{2355}{1462}\frac{3}{128\eta v^5}
 e_0^2\left(\frac{v_0}{v}\right)^{19/3}
 P_{3\rm PN}(v,v_0,\eta).
\]

The independently transcribed polynomial is stored by relative PN order in
`pn_terms()`; its 3PN term includes both \(\ln(16v^2)\) and
\(\ln(16v_0^2)\). The project waveform uses the distinct phase-origin adapter

\[
\Delta\Psi_{\rm native}(f) =
\Delta\Psi_{\rm paper}(f;f_{0,\rm ecc})-
\Delta\Psi_{\rm paper}(f_{\rm ref};f_{0,\rm ecc}).
\]

This subtraction is a native convention, not an additional term in Moore's
equation. The native positive-frequency polarization convention uses
\(\exp[-i\Psi]\), so its eccentric-to-circular complex quotient is
\(\exp[-i\Delta\Psi_{\rm native}]\).

| Component ID | Source | Variant | Status | Check |
| --- | --- | --- | --- | --- |
| `ecc.phase-3pn-e2/v1` | Moore et al. Eq. (6.26) | Paper phase, selectors 0/2/3/4/5/6 | Executable | Individual PN orders and increments against high-precision scalar |
| `ecc.phase-fref-gauge/v1` | Native phase-origin convention | Subtraction at explicit `f_ref` | Executable | Complex quotient of both polarizations with `f_ref` above, below, and equal to `f0_ecc` |
| `ecc.amplitude-adapter/v0` | Primary derivation unresolved | Optional project factor; default off | Pending | Outside this gate |

## Implementation mapping

The reference functions `pn_terms`, `phase_expr`, `evaluate`, and
`anchored_phase` live in `src/diffgw_reference/taylorf2_eccentric.py`.
`TaylorF2_Eccentric.eccentric_phase_correction` in
`src/torchwave/approximants/taylorf2_eccentric.py` and
`src/jaxwave/approximants/taylorf2_eccentric.py` are the production paths.
Their waveform evaluators resolve `f0_ref` and `f_ref_circ` separately and
subtract the eccentric term at the latter. Neither frontend calls the scalar
reference. The scalar test compares eccentric/circular waveform quotients;
the circular carrier is common to numerator and denominator and is not
requalified by this specification.

## Evidence and limitations

`tests/test_taylorf2_eccentric_math_reference.py` checks equal- and
unequal-mass systems \((1.4,1.4)\), \((1.7,1.2)\), and \((19,7)M_\odot\),
\(e_0=0.02\)–0.08, \(f_0=20\)–37 Hz and sampled positive frequencies through
82 Hz. It compares each of six PN truncations against 80-digit scalar
evaluation in both Torch and JAX CPU float64 at relative \(2\times10^{-12}\)
and absolute \(2\times10^{-11}\) rad, tests each PN increment at relative
\(3\times10^{-9}\), and checks both complex polarization quotients at
absolute \(3\times10^{-10}\). The gauge cases put `f_ref` above, below, and
equal to `f0_ecc`; none fits a phase or time shift. The strict
`--require-math-reference` inventory automatically collects this named test
module and rejects skips, expected failures, and deselection in the full
suite.

The [2026-09-27 execution receipt](../validation/taylorf2_eccentric_phase_20260927.md)
records the pinned base, backend environment, and focused test result.

This establishes sampled paper-coefficient agreement and native gauge
composition. A separate [versioned nonspinning amplitude-off waveform
slice](taylorf2-eccentric.md) has two strict same-model LAL holdouts. This
phase component alone does not establish physical accuracy,
spin-eccentric couplings, oscillatory eccentric terms, cutoff behavior,
GPU/FP32 behavior, or the optional amplitude factor.

## Reproduce the checks

With SymPy, mpmath, Torch, JAX, and pytest installed, run:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests \
  python -m pytest -q tests/test_taylorf2_eccentric_math_reference.py \
  tests/test_eccentric.py tests/parity/test_eccentric_parity.py \
  --reference-device=cpu
```

Run the full strict inventory with the [CPU qualification command](../waveforms/qualification-status.md#reproduction).
A change to the paper
coefficients, the `f0_ecc` meaning, or the native `f_ref` subtraction requires
a component version increment and renewed scalar and backend evidence.
