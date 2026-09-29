# IMRPhenomHM independent published-calibration composition — 2026-09-27

This gate compares complete complex waveforms from both native frontends with
`imrphenomhm-independent-waveform/v1`. The reference derives the D remnant,
D amplitude/phase, peak slope, HM QNM frequencies, mapped phases, PN amplitude
ratios and Wigner projection independently. It never obtains a reference input
from a Torch/JAX waveform or fit helper. Independently generated QNM numerical
assets are shared inputs; their solver and off-grid evidence remain separate.

## Configuration and scope

The native option is `calibration="published"`, with `hm_21_pn_variant="mishra_eq12b"`,
`hm_32_pn_variant="mishra_eq12d"`, and `hm_43_pn_variant="mishra_eq12g"`.
It uses the printed Khan Table V coefficients and Husa remnant coefficients,
including `0.102` for the final-spin eta*S² term. It is a named paper-precision
variant, not a recovery of historical unrounded calibration. Both
`london_2018` and `phenomd_runtime` mapping transitions are exercised.

The CPU float64 value tests use `(mass1,mass2,chi1,chi2)` of
`(35,20,0.2,-0.1)` and `(30,30,0.35,-0.2)`, distance 100 Mpc, inclination
0.93, harmonic azimuth 0.27, orbital phase 0.37, and time shift 0.008 s.
They compare all six individual complex modes and plus/cross projections,
then the full sum. Coalescence and finite `f_ref=40 Hz` gauges are tested.
Each mode samples outer mapping joins and inverse images of D amplitude/phase
joins at the join and relative offsets ±1e-6, plus Mf=0.005, 0.035 and 0.12.
An equal-mass/equal-spin case separately checks exact odd-mode zeros.

Value tolerance is `rtol=3e-8` plus `atol=2e-10` times each output's maximum
reference magnitude. Smooth frequency, inclination, harmonic azimuth, orbital
phase, time and distance derivatives are compared at Mf=0.005, 0.035 and 0.12,
finite reference frequency and both transition choices. The oracle uses
fourth-order central differences at 50-digit precision, requiring convergence
under step halving within 1e-7 relative maximum norm. Native autodiff tolerance
is `rtol=3e-6` with the same scaled absolute floor, separately per derivative.
These are sampled implementation checks; intrinsic mass/spin derivatives,
boundary derivatives, arbitrary option combinations and a continuous-domain
error bound are outside this new claim.

## Reproduction

Reviewed execution on `len`, 2026-09-27, in the source snapshot
`/tmp/diffgw-hm-source-audit-20260927/repo`, based on commit
`fd20c2988cd2b1a90b42f4a29db7f946ca5560af` plus the published-composition change:

- Full strict CPU suite: **2,144 passed**, 1,683.24 s. Its math-reference
  inventory reports **2,093/2,093 passed, zero skips/xfails and zero deselected**.
  Two existing TorchScript deprecation warnings were reported.
- Independent composition and N61 off-grid tests: **98 passed**, 186.70 s.
- Published-calibration API/constant checks: **17 passed**, 33.26 s.
- Final published API and catalog contract checks: **30 passed**, 37.91 s.
  This repeats the 17 API checks above.
- Existing D/HM regression tests: **53 passed**, 25.37 s, with one existing
  TorchScript deprecation warning.
- Final citation metadata: **13 catalog checks passed**, 4.96 s. Both generated
  documentation checks and the strict MkDocs build passed.

The environment was Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3, Torch 2.14.0,
JAX/JAXlib 0.11.2, SymPy 1.14.0, mpmath 1.3.0 and pytest 9.1.1, with
`JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu` and `PYTHONPATH=src`.
Run logs were `/tmp/diffgw-hm-independent.log`, `/tmp/diffgw-published-api.log`
`/tmp/diffgw-published-regression.log` and `/tmp/diffgw-hm-final-api.log` on len. The API file was initially
run as `test_phenomd_published_calibration.py`; it was renamed without content
changes to enter the strict math-reference inventory.

Use the maintained tests, which enter the strict no-skip CI math inventory:

```sh
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_imrphenomhm_waveform_math_reference.py \
  tests/test_phenomd_published_calibration_math_reference.py \
  tests/test_imrphenomhm_qnm_n61_offgrid_math_reference.py -ra
```

The full strict run log is `/tmp/diffgw-hm-strict-math.log` on len.
The full strict reference inventory is reproduced with:

```sh
PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest \
  tests/test_waveform_definitions.py tests/test_*math_reference*.py \
  tests/test_nsbh_*reference.py tests/test_xphm_rotation_reference.py \
  tests/test_xphm_nnlo_clean_room.py \
  --require-math-reference --reference-device=cpu -ra
```

The calibration contract additionally checks every printed fit/remnant constant,
unchanged defaults, explicit dispatcher forwarding, invalid options, unsupported
carrier/remnant overrides, batch/scalar agreement and JAX compilation.
The N61 data move into `diffgw_core.hm_qnm_n61_data` preserves coefficient payload
SHA256 `afcb21b634efacbc1ed0b44f24fda5df5ab08e236f3b7cbfdc7e71d3452d6525`;
its evaluators retain their original arithmetic.

## Relationship to earlier evidence

The earlier [intrinsic assembly receipt](imrphenomhm_intrinsic_modes_20260927.md)
remains conditional on frontend-supplied D/QNM/fit inputs. This new composition
closes that dependency for the named published option. Default
`calibration="inherited"` and binary-compatible PN settings remain unchanged.
Their existing binary and NR records do not qualify this new option.
The red 2265/3631/3619/3630 NR outcomes and bounded green 3144/2158 records stay
in the [family ledger](../waveforms/research/phenom-higher-modes.md).
No new physical-accuracy or historical LAL-equivalence claim is made here.
