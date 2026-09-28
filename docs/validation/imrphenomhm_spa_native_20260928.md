# HM SPA/SXS native convention — 2026-09-28

The opt-in `hm_phase_convention="spa_sxs_v1"` implements the
[source-derived SPA factor](../math/imrphenomhm-spa-conventions.md) and
[signed-pair transport](../math/imrphenomhm-signed-projection.md) in both native
frontends. The [canonical specification](../math/imrphenomhm-waveform.md)
contains the derivation, supported combinations and test tolerances.
Default output and all historical receipts remain unchanged.

## Mathematical implementation

Composition `imrphenomhm-independent-waveform/v3` adds two components under
`imrphenomhm-spa-sxs-carrier/v1`: `hm.waveform.spa_sxs.polarizations.v1`
and `hm.waveform.spa_sxs.derivatives.v1`. Each intrinsic mode receives
$\exp[i(m/2-1)3\pi/4]$ once before projection; both $m=2$ modes remain
bitwise unchanged. Native D is required. Either reference-phase origin and
mapping choice is supported. Applying this leading-SPA constant through
merger/ringdown is an explicit model convention, not a PN accuracy theorem.
No coefficient was fitted to NR or a binary oracle.

The independent scalar composition reuses the signed-pair symbolic factor,
independent printed D fits and existing QNM assets. The native implementations
use tensor operations and import no reference evaluator. The maintained
waveform-reference module checks all six complex modes, individual and summed
polarizations around mapped joins, both hemispheres and both phase origins.
Smooth derivatives, primary-mass/reference-frequency derivatives, batching,
JAX JIT, dispatch, masks, exchange symmetry and default preservation are also
checked. Claims cover only the explicitly tested parameter tuples; the canonical
specification records their numerical tolerances.

The focused CPU FP64 run passed **191 tests in 711.64 seconds**, with no
failures, skips or expected failures. This includes the complete waveform
module, existing signed/SPA/projection references and catalog checks. The
command below is a focused run, not the whole-project strict reference gate;
its result does not replace that gate or imply accelerator/FP32 coverage.
The new catalog entries record eleven bounded implementation claims and
preserve the eleven previous claims with explicit legacy phase convention.

Ruff passes for the changed reference, catalog and test files. Whole-file
checks of the two native modules report ten pre-existing unused-import/local
findings; comparison with `d90d697` confirms the same findings in the baseline.
After adding the scoped catalog claims, catalog tests, generated library and
coverage checks, and strict MkDocs build passed on len. Full reference CI on
the pushed revision remains a separate gate.

## Seen-NR diagnostic

This is a before/after diagnostic on already examined SXS:BBH:2374 and 2163,
not a prospective physical qualification. Both sides use
`hm_reference_phase="orbital"`; the historical published-NR receipt used
`"legacy"`. Thus the fresh baseline below is **not** a reproduction of that
historical receipt. Both use published calibration and Mishra Eq.12b/d/g PN
options, 100 solar masses, 500 Mpc, `f_ref=30 Hz`, and 30–1024 Hz at 1 Hz.
View 0 has $(\iota,\psi)=(0.7,0)$; view 1 has $(1.2,0.4)$.

The frozen manifests, raw files, conditioning, fixtures, PSD and original
metrics were hash-checked. Historical NR-resolution screens were reused only
after those identity checks; they were not recomputed. Both pass. The original
four optimizer seeds (17,23,41,67), time/orbital/polarization search, match floor
0.980, norm interval [0.85,1.15] and seed-spread ceiling $10^{-5}$ are retained.
The runtime mapping is a separate diagnostic, not a change to the frozen
published mapping. There was no new data acquisition or fit.

The table reports Torch values; JAX gives the same gate outcomes. Norm is the
PSD-weighted candidate/reference norm ratio. “Pass” here means these numerical
gates on seen data, not a new qualification claim.

| SXS | Mapping | Phase convention | View | Match | Norm | Seed spread | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2374 | London | legacy | 0 | 0.988641 | 0.937321 | 2.14e-4 | Fail: spread |
| 2374 | London | legacy | 1 | 0.971789 | 0.811100 | 5.13e-11 | Fail: match, norm |
| 2374 | London | SPA | 0 | 0.988881 | 0.956175 | 8.11e-12 | Pass |
| 2374 | London | SPA | 1 | 0.975269 | 1.112126 | 3.90e-11 | Fail: match |
| 2374 | runtime | legacy | 0 | 0.995016 | 0.962849 | 8.18e-11 | Pass |
| 2374 | runtime | legacy | 1 | 0.979583 | 0.984981 | 5.63e-6 | Fail: match |
| 2374 | runtime | SPA | 0 | 0.995329 | 0.984803 | 8.07e-12 | Pass |
| 2374 | runtime | SPA | 1 | 0.983052 | 1.143848 | 5.92e-12 | Pass |
| 2163 | London | legacy | 0 | 0.990156 | 0.976356 | 1.86e-11 | Pass |
| 2163 | London | legacy | 1 | 0.977392 | 0.824196 | 3.11e-11 | Fail: match, norm |
| 2163 | London | SPA | 0 | 0.990120 | 0.972344 | 1.13e-4 | Fail: spread |
| 2163 | London | SPA | 1 | 0.974579 | 0.941365 | 6.36e-12 | Fail: match |
| 2163 | runtime | legacy | 0 | 0.993991 | 0.976669 | 9.85e-12 | Pass |
| 2163 | runtime | legacy | 1 | 0.984505 | 0.933009 | 7.78e-11 | Pass |
| 2163 | runtime | SPA | 0 | 0.994074 | 0.999716 | 1.01e-11 | Pass |
| 2163 | runtime | SPA | 1 | 0.981391 | 1.292795 | 1.12e-5 | Fail: norm, spread |

Across all cases/options, the maximum Torch/JAX intrinsic-mode relative L2
error is $6.28\times10^{-13}$. Direct native polarizations evaluated at the
optimized orbital/time settings reproduce the independent projection within
$5.41\times10^{-16}$ relative L2. Mode magnitudes and support are preserved,
with exact equality for $m=2$. The changed interference between modes can
nevertheless change the total projected norm substantially.

The seed-spread failures remain failures. Periodic optimizer boundaries and
local optima mean these scores do not certify a global optimum. Separate
optimizer robustness and intrinsic mode-shape investigations are required;
a phase convention alone does not cure the remaining amplitude and
frequency-dependent phase discrepancies. Any later physical claim needs a
frozen model/domain and suitable untouched validation data. No existing red
receipt or threshold is reclassified.

## Execution and replay

Both tests and rescore used CPU FP64 on `len` in
`/tmp/diffgw-hm-spa-native-20260928/repo`, an archive of `d90d697` with reviewed
changes overlaid. The rescore ran 09:24:42–09:26:37 UTC with PID 811774,
returned zero, and recorded `qualification_promoted=false`. Its preserved
scratch runner, log and JSON are in the parent directory. Scratch diagnostic
files are not packaged as maintained production tools.

| Artifact | SHA-256 |
| --- | --- |
| Torch HM source | `c7e012f095e1396bc3220bcec60e665bcd8a630525e6a2f43d55b6c87ed829ef` |
| JAX HM source | `0bc379d180abc3f98d52e3bc829c644d8051853a0a6df8cc1f555f8d3b246869` |
| Independent waveform reference | `fc231f36d5c778165a5981b1817c4e9ec7e66bab484a2a7b7df85cb712842eb3` |
| Waveform-reference tests | `ffbbff1b690f43c1d115b5db1f694fe0734e569081f4de253c30339582dfce72` |
| Scratch `hm_spa_native_rescore.py` | `0426a424e853e71ccd6f6111279c6647954e32f336f1dc103b0009e9859c58bb` |
| `rescore.json` | `f3fad6cf7693ac26d4645a2f2305b9382635e9ed364ba8bbe8ac48ca8e14b659` |

The JSON also records the full native source-tree hash, frozen runners,
manifests, raw inputs, preparation, fixtures, historical metrics and PSD,
all seed outcomes, boundary flags, projection rechecks and invariants.
Original data and receipts remain under `/tmp/diffgw-hm-published-nr-20260927`.

Environment: Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3, Torch 2.14.0,
JAX/JAXlib 0.11.2, SymPy 1.14.0, mpmath 1.3.0, pytest 9.1.1,
PyCBC 2.11.0, MkDocs 1.6.1 and Ruff 0.16.8. Use
`/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin` on `PATH` on len.

```sh
export PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q \
  tests/test_imrphenomhm_waveform_math_reference.py \
  tests/test_imrphenomhm_signed_projection_math_reference.py \
  tests/test_imrphenomhm_spa_conventions_math_reference.py \
  tests/test_imrphenomhm_projection_math_reference.py \
  tests/test_waveform_definitions.py
python scripts/generate_waveform_library_table.py --check
python scripts/generate_waveform_coverage.py --check
python -m mkdocs build --strict
```

To replay the scratch NR diagnostic with the same pinned inputs and a new
output path, use the environment above and:

```sh
cd /tmp/diffgw-hm-spa-native-20260928/repo
python ../hm_spa_native_rescore.py --repo "$PWD" \
  --source-revision YOUR_CHECKOUT --output /tmp/new-hm-spa-rescore.json
```

The runner rejects modified historical inputs and existing output paths. It
records the current source-tree hash; it does not itself freeze a new
qualification protocol.
