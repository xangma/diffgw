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
The full arXiv v2 source archive was checked before changing production. Its
SHA-256 is `cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`;
it contains the author amplitude/phase notebooks, 37 amplitude `.m` tables
(`21:10`, `32:9`, `33:9`, `44:9`), and 33 phase `.m` tables
(`21:7`, `32:12`, `33:7`, `44:7`), eight mixing-coefficient files, and the fit
README. Its README describes those as the fits for
the paper implementation, but the archive has no release manifest tying them
to default release `122022`, its all-new amplitude option, or a complete
current-release collocation provider. Existing checksum-pinned fixtures verify
selected paper-era surfaces only. Paper-era `(3,3)` files are present; their
mapping to the **recalibrated 122022** provider, together with the other
modes' current-default amplitude calibration, remains unresolved. No complete
standalone XHM 122022 calibration asset is registered in
[external data](../external-data.md).

Consequently, routing the existing earlier-provider nodes, compatibility sign
branch, or older transition prescription into production would create an
unvalidated hybrid. A justified fix requires an author-published, checksum-
pinned current-release fit asset or an independently reproducible calibration
dataset and procedure. Neither is available here, so this change makes no
production waveform modification.

### Paper-era versus release-era boundary

The author-published arXiv v2 archive is a coherent paper-era asset: its
checksum-pinned 218-member source bundle contains 37 amplitude-fit files, 33
phase-fit files, 8 complex mixing-coefficient files, 10 QNM files, and the
author notebooks. Appendix A, Eqs. (55)--(57), publishes the geometric
spheroidal-to-spherical transform for `(2,2)` and `(3,2)` and states that its
mixing coefficients depend only on final spin. These sources are sufficient
to define a clean paper-era mixing reference; they are not evidence that a
current-release `(3,2)` spheroidal amplitude is the same object.

The official
[LALSuite 7.26 API documentation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_i_m_r_phenom_x__c.html)
draws the release boundary more sharply. It identifies `122022` as the default
after a higher-mode **amplitude
recalibration**, retains `122019` as the old release, separates amplitude
region/collocation frequencies, fit surfaces, and ansatz versions, and states
that `122022` did not modify phases. The corresponding author merge request
[!2128](https://git.ligo.org/lscsoft/lalsuite/-/merge_requests/2128), merged
2023-04-06 at commit
[`9eaa3722`](https://git.ligo.org/lscsoft/lalsuite/-/commit/9eaa372236e5a5d51155dad17a4f53d7255285f5),
says the recalibration propagates to XPHM. Its linked review/calibration wiki
redirects unauthenticated readers to sign-in and no checksum-pinnable 122022
calibration deposit was found in this bounded search.

Therefore the exact missing public fields are:

1. release-122022 amplitude fit surfaces or coefficients for inspiral,
   intermediate, and ringdown collocation values for `(2,1)`, `(3,2)`,
   `(3,3)`, and `(4,4)`;
2. their region-cut and collocation-frequency maps, veto/branch rules, and
   amplitude-ansatz version mapping;
3. a hashed, licensed manifest tying those assets to release `122022`; and
4. for the complete current `(3,2)` mode, the recalibrated spheroidal
   amplitude inputs used before the published basis transform.

The paper-era phase equations, phase assets, and mixing geometry remain the
best public map because the release notes say phases were unchanged, but no
public release manifest proves file-for-file identity. During this audit a
broad web search unexpectedly exposed generated-source search snippets. No
snippet, LAL source formula, coefficient, or implementation detail was used;
the mapping above comes only from the papers, author archive, official API
documentation, public merge metadata, and existing binary-output receipts.
The arXiv page identifies its manuscript/source distribution license, while
the inspected fit-asset README gives no separate redistribution terms for the
tables. No accessible `122022` calibration deposit with version and reuse
terms was found.
Any future clean-room implementation of the 122022 delta must start in a
fresh context with an independently published calibration asset.

The separate [release-boundary diagnostic](../validation/xhm_release_boundary_20260927.json)
checks the public `122019` and `122022` binary interfaces on both frozen
parameter cases. The `(2,1)`, `(3,3)`, and `(4,4)` spherical phases, and the
`(3,2)` **spheroidal** phase, are bitwise equal on its grid. The `(3,2)`
**spherical** phase is not: the recalibrated spheroidal amplitude enters the
published Appendix-A mixing transform. After removing one reference-phase
offset, its wrapped phase-shape RMS changes by `0.1771` and `0.03565` radians
for the catalog and holdout cases. Thus “phases unchanged” identifies the
underlying phase prescription, not invariance of the observable mixed mode.
This is a binary-output diagnosis only; it supplies no missing amplitude fit.

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

The exact PR-base qualification rerun at
`4814835f7744fe00a5863ed70eb590d41a7e3ac3` is recorded in the
[release-qualification receipt](../validation/xhm_xphm_release_qualification_20260925.json).
It reproduced the `(3,2)` inspiral-amplitude relative L2 values
`0.47847072820479275` (catalog) and `0.10594593969797324` (holdout). The
strict whole-complex-mode holdout also fails: direct complex relative L2 is
`1.5613045346149474` and flat mismatch is `0.617904191209656`, against
`1e-6` and `1e-8` limits. The qualification verdict is therefore no-release,
with no production waveform, transition, or gate change.

## Reproduce the checks

Run the paper/scalar/frontend checks with:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src \
  pytest -q tests/test_xhm_inspiral_amplitude_math_reference.py
```

The regional decomposition above is a historical, one-off binary diagnostic;
its replay script is not maintained. With the pinned optional LAL Python
bindings installed, run the maintained frozen binary gate and clean inspiral
assembly diagnostic with:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src \
  python scripts/qualify_xhm_xphm_frozen_binary.py

PYTHONPATH=src python scripts/diagnose_xhm_inspiral_assembly_clean.py
```
