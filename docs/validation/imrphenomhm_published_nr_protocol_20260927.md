# Published-calibration IMRPhenomHM: prospective two-case NR protocol

Freeze this protocol, its two JSON manifests, runner and data-free tests in Git
before retrieving or decoding either selected strain. The target is the exact
published-calibration option below. Preserve every failure or infeasible result;
do not replace a case, alter conditioning, tune coefficients or relax a gate after
inspection. Earlier default-calibration receipts remain unchanged.

## Target and literature

Use native Torch and JAX IMRPhenomHM with all of:

```python
calibration="published"
hm_21_pn_variant="mishra_eq12b"
hm_32_pn_variant="mishra_eq12d"
hm_43_pn_variant="mishra_eq12g"
hm_mapping_transitions="london_2018"
```

The [independent mathematical specification](../math/imrphenomhm-waveform.md)
defines the printed Khan/Husa calibration and independently generated N61 QNM
inputs. This is a named source-based variant; it does not claim to reproduce
unrounded historical calibration or published population accuracy exactly.

[London et al., Results/Fig. 3](https://arxiv.org/html/1708.00404v4) motivates
the full-NR signal through ell=5, Advanced LIGO design PSD, 100-solar-mass
comparison, 30-Hz floor, fixed intrinsic parameters and inclination, and
template time/orbital-phase/polarization optimization.
[SXS 2019 Appendix B](https://arxiv.org/html/1904.04831#S.app2) motivates one
common time/orbital rotation across NR modes in a plus/cross network.
[SXS Catalog 3](https://arxiv.org/html/2505.13378v2) supplies versioned strain
and conventions. The two views, taper and thresholds here are project choices,
not published guarantees or a reproduction of a paper's population average.

## Metadata-only selection

Use official [sxscatalogdata v3.0.0 simulations.json](https://raw.githubusercontent.com/sxs-collaboration/sxscatalogdata/v3.0.0/simulations.json),
32,618,880 bytes, SHA-256
`132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6`.
Only catalog metadata and file manifests were inspected for this selection.

Reuse the exact nonspinning filter/ranking in the
[3144 protocol](imrphenomhm_sxs_3144_prospective_protocol_20260927.md) and
aligned-spin filter/ranking in the
[2158 protocol](imrphenomhm_sxs_2158_prospective_protocol_20260927.md).
Additionally exclude every explicit SXS ID found in tracked repository text:
0001, 0002, 0008, 0030, 0056, 0063, 0166, 0169, 0182, 0185, 1166, 2158,
2265, 2331, 2483, 2484, 2485, 2486, 2489, 2491, 2707, 3144, 3619, 3630,
3631, 9997, 9998, 9999. Metadata-only candidate-list mentions are not prior
strain inspection. The remaining nonspinning rank is 2374, 2499, 2164, 2498;
the aligned-spin rank is 2163, 2162, 2155, 2502. Select the first in each.

| Case | Reference q | Model chi1, chi2 | Highest/lower resolution | Reference time M |
| --- | ---: | --- | --- | ---: |
| [SXS:BBH:2374v3.0](https://doi.org/10.26138/SXS:BBH:2374v3.0) | 4.999912483619435 | 0, 0 | Lev3 / Lev2 | 570.5 |
| [SXS:BBH:2163v3.0](https://doi.org/10.26138/SXS:BBH:2163v3.0) | 3.0000780199258803 | 0.5999108664616073, 0.6000124147825587 | Lev4 / Lev3 | 1379.5 |

2374 has eccentricity 3.26e-5 and 22.055961932088238 post-reference orbits;
2163 has eccentricity 0.0001502 and 23.187710232835745 post-reference orbits.
Their run dates are 2018-09-21 and 2018-12-04. Their alternative identifiers
are `BFI:Rerun2013Catalog:0055` and `BFI:RerunAlignedCatalog:0291`.
They are unseen local prospective holdouts, but these rerun identities mean
independence from every historical model-development dataset is unproven.
No inference of a uniformly validated mass-ratio/spin domain follows from them.

The machine-readable manifests
[2374](imrphenomhm_published_sxs_2374_protocol_20260927.json) and
[2163](imrphenomhm_published_sxs_2163_protocol_20260927.json) bind exact reference
spin vectors, record IDs, six official file identities per case, levels and
this protocol's hash. Retrieve only those listed files from CaltechDATA
after the freeze. Verify official size/MD5 before decoding and record SHA-256.
Raw and conditioned arrays remain outside Git; dataset terms remain separate
from the software license. Preserve v3 memory/center-of-mass modifications.

## Frozen conditioning and comparison

Apply the complete fixed conditioning, PSD, projection, NR alignment and
model optimizer of the [3144 protocol](imrphenomhm_sxs_3144_prospective_protocol_20260927.md),
substituting each manifest's real highest/lower levels, q, spins and reference
time, and the exact published options above. The reusable runner must retain
real level labels in source records and disclose any internal role aliases.

In particular: full inertial NR modes ell=2..5 including m=0; 100 solar masses;
500 Mpc; f_ref=30 Hz; exact catalog q; common post-reference interval; cubic
complex interpolation at 8192 Hz; 500 M rise and 100 M fall; direct Fourier
sum at integer 30..1024 Hz; aLIGOZeroDetHighPower PSD; views (iota,psi)=(0.7,0)
and (1.2,0.4). The nonspinning model sets both spins to zero; its tiny catalog
spins are residuals. The aligned model uses exact reference z-spins and does
not fit the negligible transverse components.

For NR resolution, choose one orbital phase and one arrival time jointly across
all modes and both plus/cross views, using the 2048-point periodic phase grid
and eight-peak refinement. Apply the same optimum to both view screens. For
each model/backend/view, optimize only arrival time, orbital phase and
polarization with seeds (17,23,41,67), popsize=12, maxiter=120, tol=1e-9 and
polish=True. Intrinsics, inclination, distance and all amplitudes remain fixed.
Time shifts use k/8192 s for integer k=-819..819. No mode-specific alignment
or amplitude correction is allowed.

## Acceptance and reporting

Each resolution view requires match >=0.995 and norm ratio in [0.95,1.05]
under the same common alignment. Each native backend at each model view
requires optimized match >=0.980, norm ratio in [0.85,1.15], and four-seed
match spread <=1e-5. These unchanged project thresholds separately constrain
shape and amplitude. A failed/infeasible resolution screen prevents a physical
qualification claim even when model scores pass. Record cases individually;
a two-case pass requires every case, backend, view and resolution gate to pass.

Validate direct/FFT agreement, mode-basis reconstruction, manifest and source
hashes, and exact option forwarding before scoring. Record all optimizer seeds,
norms, source/fixture/PSD hashes, package versions and full-minus-six-mode NR
diagnostics. No production coefficients or mathematical definitions change.
A pass is limited to these cases, mass, PSD, views and declared options; a
failure is retained with its measured cause, without rescoring as a new pass.

## Execution

Use `scripts/run_imrphenomhm_published_nr.py` with each frozen manifest and
an external data directory on `len`. Prepare with the current SXS environment;
evaluate with the current compatible Torch/JAX/PyCBC environment. Record the
freeze commit and exact commands in the subsequent receipt. The runner and
both manifests must be committed before either preparation command begins.
