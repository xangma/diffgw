# HM orbital reference-phase implementation — 2026-09-28

The opt-in `hm_reference_phase="orbital"` restores orbital covariance of the
reference-frequency origin. It is a project convention derived from London's
phase composition and the orbital rotation law, not a recovered historical
HM setting. Default `legacy` behavior and all prior evidence remain unchanged.
The [canonical specification](../math/imrphenomhm-waveform.md) gives the
source equations, derivation, component IDs and support restrictions.

## Scope

Composition `imrphenomhm-independent-waveform/v2` reuses the original v1
components and adds `imrphenomhm-orbital-reference-phase/v1`. Native Torch and
JAX omit only the extra legacy factor `exp[i(m/2-1)*s*Msec*f_ref]`. The independent
scalar evaluator follows the same declared convention using its separately
transcribed published D calibration, PN expressions and mode maps.

All six complex modes and both polarizations are compared in CPU float64 with
50-digit scalar evaluation. The independent implementation claims use published
calibration and Mishra Eq.12b/d/g PN options only. Inherited calibration has
API/default-preservation checks, not a new independent composition claim.
Both `london_2018` and `phenomd_runtime` mapping choices are checked.

The maintained `tests/test_imrphenomhm_waveform_math_reference.py` covers:

- Complete mode and polarization values for `(35,20,.2,-.1)`, distance 100 Mpc,
  `f_ref=40 Hz`, existing branch/join samples, both mapping choices. Value gate:
  `rtol=3e-8`, absolute floor `2e-10` times the comparison-array scale.
- Independently predicted orbital covariance for that binary and
  `(30,30,.35,-.2)`, `Mf=.005,.025,.06`, reference changes None→30, None→40,
  and 30→40 Hz. Compensation is `phic += (C_after-C_before)/2`, with
  `C=pi/4-phi(Msec*f_ref)+s*Msec*f_ref`; relative gate `2e-10` with the same
  scaled absolute floor. Mapped phases and continuity anchors must remain
  below the D cutoff, `Mf=.2`.
- Smooth `f_ref` and primary-mass derivatives of all six modes and summed
  polarizations for the first binary, 130 Hz, reference 40 Hz, London mapping.
  Fourth-order scalar differences converge under step halving within `1e-7`
  relative maximum norm. Native autodiff uses `rtol=3e-6` and a column-scaled
  `2e-10` absolute floor. Other intrinsic derivatives remain outside this claim.
- Independent batch/scalar values, vector reference frequencies, JAX JIT and
  public dispatch; exact default/legacy, no-reference and m=2 reductions;
  reference-alias precedence and invalid/unsupported-carrier rejection.

Orbital requires the exact native D carrier. Explicit reference-frequency
aliases require an explicit `f_ref`; silently choosing a different first grid
sample in each mapped carrier would not define a common orbital origin.

## Execution

Reviewed tests ran on `len` in
`/tmp/diffgw-hm-orbital-20260928/repo`, based on commit
`6dc1f78d7d1894e6d69b19ecae4a13ca0fe027c6` plus this change.

```sh
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_imrphenomhm_waveform_math_reference.py \
  tests/test_phenomd_published_calibration_math_reference.py \
  tests/test_waveform_definitions.py --reference-device=cpu -ra
```

Across the initial run and targeted reruns, all **81 distinct focused checks**
passed. The initial 79-test run completed in 399.91 s with 77 passes and two
alias-equality failures. Both failures were FP64 roundoff (maximum relative
error 2.71e-16), caused by extracting a carrier magnitude under an explicit
phase alias. Only the new alias-equivalence assertion changed to `8*eps`
relative tolerance with zero absolute tolerance; its mode outputs are now
checked too. The independent value/derivative/covariance thresholds and
bitwise default/legacy reductions were not changed.

The final catalog/alias/unsupported-carrier rerun passed 17 tests in 25.82 s.
A separate final catalog/batch/JIT/dispatch/unsupported-carrier run passed
17 tests in 38.52 s. These overlap the initial run; they are not 34 additional
independent checks. The two unsupported-carrier tests were added after initial
collection and passed separately as well. Native formulas and independent
reference expressions were unchanged between these runs.

Environment: Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3, Torch 2.14.0,
JAX/JAXlib 0.11.2, SymPy 1.14.0, mpmath 1.3.0, pytest 9.1.1.
Logs on len are `tests.log`, `final-contract.log`, and
`final-alias-contract.log` under `/tmp/diffgw-hm-orbital-20260928/`;
`tests.xml` retains the initial failures rather than overwriting them.
Generated catalog documentation and its `--check`, Ruff for the changed
catalog/reference/test files, and strict MkDocs passed.

The final implementation and maintained test file identities are:

| File | SHA-256 |
| --- | --- |
| `src/torchwave/approximants/imrphenom_hm.py` | `2add1c63a3627cedf52c8dd5583a6a4ccbdd25de9569b26564c22924ea4ef75f` |
| `src/jaxwave/approximants/imrphenom_hm.py` | `5e2028dd5411bba4daf7786e9271fcc9c7c619d60d056af24804ca8e213bfdb6` |
| `src/diffgw_reference/imrphenomhm_waveform.py` | `831512bc4d3f5a3c2b84255e6723deea342a6e01ed24c76cc1a1ad800abdf6c9` |
| `tests/test_imrphenomhm_waveform_math_reference.py` | `d8bce7d44026f64a7b280bea300e801217579b9c68de11873741ac7f692e8b1a` |

The tests remain in the existing strict CI math-reference inventory. The focused
command above exercises this change and its neighboring contracts; it is not
itself a rerun of the entire strict inventory.

## Physical accuracy remains separate

The [mapping diagnostic](imrphenomhm_orbital_mapping_20260928.md) uses the new
native option against the already examined 2374/2163 fixtures and reproduces
the prior independently corrected scores. The published London option remains
below the frozen match/norm gates in the second view. Phase-map choices dominate
the observed improvement after reoptimization; amplitude-map changes alone
have little effect. No diagnostic hybrid is exposed as a new production option.
Neither these seen-fixture comparisons nor the implementation tests qualify
whole-family physical accuracy or change any earlier red gate.
