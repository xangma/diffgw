# IMRPhenomXHM higher-mode amplitude closure

## Scope and status

This component closes only the printed post-Newtonian baseline of the XHM
higher-mode inspiral amplitude. Its stable IDs are
`xhm.amplitude.inspiral.appendix-e.v1` and
`xhm.amplitude.inspiral.pn-normalized.v1`. An independent scalar reference,
the clean Torch complex implementation, and the production Torch and JAX
magnitudes agree for modes `(2,1)`, `(3,2)`, `(3,3)`, and `(4,4)`.

This is not a complete inspiral amplitude model. It excludes fitted
collocation surfaces, collocation vetoes, release-dependent transition cuts,
intermediate and ringdown calibration, and `(3,2)` mode mixing. It therefore
does not promote XHM or XPHM.

## Inputs, conventions, and options

The scalar reference accepts a positive dimensionless frequency `Mf`,
`eta`, positive-mass-ordering `delta`, and the aligned-spin combinations
`chi_s` and `chi_a`. It supports only the four displayed positive-`m` modes
listed below. It intentionally exposes no release, fit-provider, cutoff, veto,
or mode-mixing option.

## Mathematics and components

The source is García-Quirós et al.,
[arXiv:2001.10914v2](https://arxiv.org/html/2001.10914v2), Sec. IV A and
Appendix E. With

\[
  v_m = \left(\frac{2\pi M f}{m}\right)^{1/3},
\]

the scalar reference transcribes every displayed term through `v_m^6` in the
Appendix-E expressions for `Hhat_21`, `Hhat_32`, `Hhat_33`, and `Hhat_44`. The
normalized PN baseline is

\[
  \mathcal H^{\rm PN}_{\ell m}(f)
  = \left(\frac m2\right)^{7/6}
    \left|\widehat H_{\ell m}(v_m)\right|.
\]

Writing `u = f / f_ins`, the paper's complete inspiral ansatz adds three
pseudo-PN terms,

\[
  \mathcal H_{\ell m}(f)
  = \mathcal H^{\rm PN}_{\ell m}(f)
  + \alpha_{\ell m} u^{7/3}
  + \beta_{\ell m} u^{8/3}
  + \gamma_{\ell m} u^3,
\]

whose coefficients are fixed from fitted values at
`[0.5, 0.75, 1.0] f_ins`. The printed equations specify the solve after those
values are supplied; they do not publish a complete current-release
parameter-space calibration.

## Implementation mapping

| Component | Scalar | Torch | JAX | Claim |
| --- | --- | --- | --- | --- |
| complex Appendix-E polynomial | `printed_hhat` | `hhat_lm_printed` | magnitude only | printed equations |
| normalized PN magnitude | `normalized_pn_amplitude` | production `_pn_amplitude_ratio` | production `_pn_amplitude_ratio` | printed equations |
| pseudo-PN collocation solve | equation helper | clean helper | not routed | supplied nodes only |
| fitted current-release nodes | unavailable | inherited native surfaces | inherited native surfaces | no independent asset |

The scalar module has no Torch, JAX, LAL, fit-table, or waveform-runtime
dependency. Frozen complex checkpoints retain phase-sensitive information
before either production implementation takes the magnitude. The frontend
test also covers Torch frequency gradients and JAX JIT.

## Evidence and limitations

### Earliest mismatch

The binary-only regional diagnostic evaluates public installed LAL mode
outputs without inspecting source or using their values as fit targets. On the
exact stacked base `576f325f8da292ecb550ffc7ebde90e71399bc07`, the earliest
material discrepancy is higher-mode **inspiral amplitude**, and `(3,2)` is
worst in both frozen cases:

| Mode | Catalog inspiral amplitude relative L2 | Holdout inspiral amplitude relative L2 |
| --- | ---: | ---: |
| `(2,1)` | 0.21601 | 0.02318 |
| `(3,2)` | **0.47847** | **0.10595** |
| `(3,3)` | 0.17294 | 0.08986 |
| `(4,4)` | 0.39970 | 0.08534 |

By contrast, higher-mode inspiral phase RMS after an independent affine
phase/time gauge removal is `4.54e-5`--`1.33e-4` rad across these rows. This
does not qualify the complete phase: intermediate phase residuals are
`6.24`--`28.80` rad, while intermediate amplitude relative L2 errors are
`0.77`--`7.04`. Ringdown residuals also remain nonzero, and the finite 512 Hz
grid does not reach the catalog `(4,4)` ringdown region. Exact values and
environment versions are in the
[component receipt](../validation/xhm_carrier_components_20260925.json).

### Why production is unchanged

The new scalar checks show that both production frontends already implement
the printed Appendix-E baseline. Changing those coefficients would contradict
the cited equations and would not supply the missing fitted collocation data.
The maintained public-API diagnostic also distinguishes the installed default
release `122022` from the explicit author-era `122019` control; their amplitude
outputs are not interchangeable. A 20-row clean-assembly run records result
SHA-256 `57230acd...1826` and raw-array SHA-256 `a4379774...0f32` in the
[component receipt](../validation/xhm_carrier_components_20260925.json).
Existing checksum-pinned author-archive fixtures verify selected paper-era
amplitude surfaces from the arXiv v2 `AmpFits` supplement, but neither that
selection nor the archive inventory identifies a complete current-release
node provider. In particular, the `(3,3)` provider calibration and the
current-default all-new amplitude calibration remain unresolved. No complete
standalone XHM 122022 calibration asset is registered in
[external data](../external-data.md).

Consequently, routing the existing earlier-provider nodes, compatibility sign
branch, or older transition prescription into production would create an
unvalidated hybrid. A justified fix requires an author-published, checksum-
pinned current-release fit asset or an independently reproducible calibration
dataset and procedure. Neither is available here, so this change makes no
production waveform modification.

### Intermediate, ringdown, and native calibration boundary

The paper supplies analytic ansätze for intermediate and ringdown regions,
but their mode-dependent coefficients are calibrated to hybrid/NR data. The
native implementation intentionally uses its independently maintained SXS fit
surfaces and native Kerr-Leaver QNM frequencies. Those choices are physical
calibration paths, not numerical reconstructions of the installed public LAL
release; agreement with the printed functional form therefore does not imply
binary parity. Mode `(3,2)` additionally requires spherical-spheroidal mixing.

The component receipt is descriptive evidence, not a new acceptance gate.
The strict frozen full-waveform catalog/holdout results remain `0/1` for XHM
and `0/1` for XPHM in both sets, hence red.

## Reproduce the checks

Run the paper/scalar/frontend checks with:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src \
  pytest -q tests/test_xhm_inspiral_amplitude_reference.py
```

With the pinned optional LAL Python bindings installed, reproduce the
binary-only regional and release-control evidence with:

```bash
PYTHONPATH=src DIFFGW_GIT_REVISION=$(git rev-parse HEAD) \
  python scripts/diagnose_xhm_frozen_components.py \
  --output /tmp/diffgw-xhm-frozen-components.json

PYTHONPATH=src python scripts/diagnose_xhm_inspiral_assembly_clean.py
```
