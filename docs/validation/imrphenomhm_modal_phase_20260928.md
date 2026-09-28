# HM modal phase: intrinsic NR diagnostic

The two seen NR cases contain a substantial relative-mode phase residual after the orbital-reference
correction. Between the London and runtime phase joins, the runtime mapping leaves an approximately constant
primitive-invariant offset of −2.41 to −2.56 rad in 33 and 44. The London mapping develops a larger positive
slope there. Neither mapping removes the later frequency dependence. These differences exceed the NR
resolution differences in the same bands. This localizes a scientific question; it does **not** establish a
source correction, select a production mapping, or promote qualification.

This follows the [amplitude/phase mapping diagnostic](imrphenomhm_orbital_mapping_20260928.md). Mass, spin,
distance, published calibration, Mishra PN options, 30–1024 Hz fixture, PSD and conditioning remain unchanged.
Native modes use `hm_reference_phase="orbital"` and `f_ref=30 Hz`. Both Torch and JAX, all six model modes,
and both retained NR resolutions are evaluated: 2374 Lev3/Lev2 and 2163 Lev4/Lev3. No new holdout is consumed.

## Observable and controls

The retained signed-mode polarization basis gives the intrinsic positive-frequency quantity

\[
P_{\ell m}=(h_+ + i h_\times)/\overline{{}_{-2}Y_{\ell m}},
\qquad u_{\ell m}=P_{\ell m}/|P_{\ell m}|.
\]

The signed harmonic is retained; there is no factor of two. Reconstruction agrees between both inclination
views to relative PSD-weighted L2 below 3.78e-16. With $g=\gcd(m,2)$, the primitive invariant is

\[
T_{\ell m}(f)=u_{\ell m}(f)^{2/g}
  \overline{u_{22}(2f/m)}^{m/g},\qquad
R=T_{\rm native}\overline{T_{\rm NR}}.
\]

Common time and orbital phase cancel exactly. For 33 the powers are 2 and 3; for 44 they are 1 and 2,
preserving a possible 44 sign error that squaring would hide. The output also retains the unreduced invariant.
The displayed slope is $(g/2)d\arg R/df$, in rad/Hz, so the two modes have comparable relative-mode slope
units. Phase statistics are unweighted over retained bins; PSD weights apply to power fractions and alignment.

Direct modal residuals supplement the invariant. One 22-only PSD-weighted delay search, followed by bounded
scalar refinement within adjacent time-grid cells and analytic orbital phase, aligns all six modes together.
Both global orbital branches $\alpha$ and $\alpha+\pi$ are retained; no branch is selected independently for
an odd mode or frequency bin. There is no per-mode fit. The 22 invariant is identically one and is not
evidence of carrier accuracy; the direct 22 control is:

| Case / resolution | 22 match after common alignment | Native / NR norm | Direct phase RMS, rad |
| --- | ---: | ---: | ---: |
| 2374 / Lev3 | 0.999306242 | 1.006536 | 0.055873 |
| 2374 / Lev2 | 0.999324627 | 1.006536 | 0.055857 |
| 2163 / Lev4 | 0.999390972 | 1.010397 | 0.143914 |
| 2163 / Lev3 | 0.999400249 | 1.010345 | 0.137940 |

No 22 time optimum lies at the search boundary. London/runtime 22 outputs agree to relative complex L2 at most
1.205e-15. Maximum Torch/JAX relative complex mode L2 is 6.273e-13 across both cases and mappings.

## Where the discrepancy appears

`L` is `london_2018`; `R` is `phenomd_runtime`. The table uses Torch and the higher NR resolution. Phase means
are circular means of $R$, without dividing the phase by its exponent; slopes include the divisor defined
above.

| Case / mode | London / runtime join, Hz | Retained bins between joins | Mean phase L / R, rad | Median slope L / R, rad/Hz |
| --- | --- | --- | --- | --- |
| 2374 / 33 | 45.47 / 58.46 | 48–57, 4 bins | −2.4106 / −2.5630 | +0.013870 / −0.001622 |
| 2374 / 44 | 61.55 / 79.14 | 62–78, 9 bins | −2.3508 / −2.4571 | +0.016492 / −0.001120 |
| 2163 / 33 | 44.64 / 57.39 | 45–57, 5 bins | −2.4212 / −2.5557 | +0.012381 / −0.002314 |
| 2163 / 44 | 60.39 / 77.65 | 62–76, 8 bins | −2.3009 / −2.4076 | +0.017384 / −0.000734 |

The corresponding high/low NR invariant phase RMS values are 0.000773, 0.000536, 0.001699 and 0.000717 rad.
Changing NR resolution changes each reported circular phase mean by less than 0.000734 rad. This supports a
resolved model/convention discrepancy in these bands, rather than an NR resolution explanation. It is not an
exhaustive NR error estimate.

The offset remains frequency dependent. Above both joins and below 0.9 times the mode ringdown frequency,
runtime median relative slopes are +0.00397 to +0.00621 rad/Hz; London slopes are +0.00467 to +0.00667 rad/Hz.
Runtime ringdown-window circular means are −0.9415, −0.8800, −0.1878 and −0.0159 rad in table order. Maximum
adjacent invariant-phase increment is 0.108 rad, below the pi/2 unwrap warning. Unwrapping restarts across
unsupported gaps.

A common complex strain/Fourier convention $c=e^{i\gamma}$ multiplies both primitive 33 and 44 invariants by
$c^{-1}$. Similar offsets therefore need not be independent modal defects; the later drift cannot be removed
by one constant. The invariant also retains carrier error at its mapped frequency. The [London
construction](https://arxiv.org/html/1708.00404v4) and independent reference enforce phase/first-derivative
continuity: changing a join changes curvature and accumulated phase, not necessarily continuity.

**Next scientific action:** audit the source-backed common Fourier/strain convention and stationary-phase
modal offsets, including primitive 33/44 low-frequency limits. Derive a convention before proposing a
correction; separate its constant contribution from mapping/carrier drift. Do not fit a constant or select a
production mapping using these seen NR scores.

**Follow-up:** the [SPA and raw-TD convention diagnostic](imrphenomhm_spa_conventions_20260928.md)
derives a fixed branch conversion and checks it on these same cases. It accounts for most of the
transition-band constant but leaves later drift. Signed-mode projection and gauge transport remain
necessary before interpreting the factor as a production correction.

## Coverage and limitations

Exact mapped bins restrict 33 to multiples of 3 Hz starting at 45 Hz, and 44 to even bins starting at 60 Hz.
Below those bounds lie respectively 18.51% / 23.55% of 2374's 33 / 44 PSD-weighted power and 12.10% / 17.26%
for 2163. Thus the invariant cannot establish a broad pre-London-join plateau. Direct common-22 residuals
cover lower frequencies. Sampled-power fractions also reflect decimation; they are not frequency-region
coverage.

Bins require amplitude at least 0.001 of each mode's peak in both NR resolutions and native mappings. Direct
masks retain over 99.9999% of 33/44 NR power. Across full invariant support, high/low NR phase RMS rises to
0.0113–0.0219 rad. This mask does not prove phase accuracy. Raw direct high/low NR differences retain separate
time/orbital origins and are not intrinsic error estimates; the invariant removes those origins without fits.

Opposite-chirality power is at most 3.65e-8 for 33/44, reaching 9.17e-5 in 2374's 21 mode. Negative-m symmetry
residual reaches relative L2 2.19e-4 across six modes; exact symmetry is not imposed on NR. Magnitude error
also remains: 33 native/NR norms are about 1.08, and 44 norms about 1.21 / 1.36 for 2374 / 2163. This
diagnostic does not close amplitude or family validation.

## Evidence and replay

The completed run used `len`, PID 968546, 2026-09-28 01:43:59–01:44:23 UTC. Python 3.12.3, NumPy 2.3.5, SciPy
1.16.3, PyCBC 2.11.0, Torch 2.14.0, JAX/JAXlib 0.11.2 and SymPy 1.14.0 supplied CPU FP64 evaluation. The
script verifies the frozen manifests/source, raw hashes, sidecars, fixtures and PSD, then records the distinct
native source identity.

| Artifact / input | SHA-256 |
| --- | --- |
| `len:/tmp/diffgw-hm-phase-20260928/hm_modal_phase_diagnostic.py` | `0df7b84c019fc54b5b7b9c5fb127f585f8c15201f9273f9a15a00ef0a3dff7cb` |
| `len:/tmp/diffgw-hm-phase-20260928/modal_phase_v3.json` | `8522d7256181f5014b4f4dd3010a3959324043579177f9de0e2ed107993e0d81` |
| Native helper `hm_orbital_mapping_ablation.py` | `fc9e67aabd89275a6d28710bb8f98196cc6a7bbaad8b3f544e49aa2ccd65264e` |
| Native Python source tree | `4fbc0f66c70c78e79da10cff29ab380b02d14c5518f7d7d879f09aeda1a6be89` |
| Frozen Python source tree | `15d843ade209bbd36a65b8c1b850a415182d4296ab3994dd1dcdd304a9cd446a` |
| Frozen / native protocol runner | `1cba5bb618c1994cf4851c6dbcf3a2a7c46b95d9ccc3706805432b5e297c6be2` |
| Frozen / native base runner | `89856b9b94af8cf546620d2f07928016a35f8ddafa4277e9c479b0638d18b2c0` |
| 2374 manifest | `4381ddf5537e328c0708ce9f77152ea985c075302a5d7f1f09e62fd2cb13fccd` |
| 2374 fixture | `e163d466ac0f56f6506d09ce438b4c52cd919fefab7197884038e038f085d0b1` |
| 2163 manifest | `477083ba0fa8ba6346008d946cd41504abc1111d9f727e82ff8bdbc42ed80b58` |
| 2163 fixture | `bb6cc4dc43f726aca4c9209c76d7ba615167aaa89e9fe06e6bbe841a366ae707` |

Two preliminary attempts aborted: an overly strict bitwise 22 equality check rejected floating-point roundoff,
then its replacement exposed a diagnostic variable-order error. The completed v3 records the measured relative
22 difference and requires it to be at most 32 machine epsilons. These were scratch diagnostic repairs;
neither native waveform code nor qualification thresholds changed. Only the completed artifact supplies the
results above.

Replay with the hash-recorded scratch script and native snapshot into a new external output path:

```sh
cd /tmp/diffgw-hm-orbital-20260928/repo
export PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  /tmp/diffgw-hm-phase-20260928/hm_modal_phase_diagnostic.py \
  --output /tmp/diffgw-hm-phase-20260928/modal_phase_replay.json
```

The script reads frozen source/data under `/tmp/diffgw-hm-published-nr-20260927/` and the pinned native helper
under `/tmp/diffgw-hm-orbital-20260928/`. Scratch code and large curve outputs remain external.
`qualification_promoted` is false.
