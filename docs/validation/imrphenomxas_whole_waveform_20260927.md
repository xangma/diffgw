# IMRPhenomXAS complex-polarization binary gate — 27 September 2026

**Result:** The maintained FP64 Torch/JAX comparison passes 6/6 cases on
`len`, with no skips or alignment. Across the three frozen systems, both
complex polarizations agree with the pinned public LALSimulation binary on
every nonzero binary sample at relative complex $L_2 < 1.89\times10^{-13}$,
below the predeclared $10^{-10}$ gate. This qualifies the sampled, assembled
carrier on the public waveform's support; it does not certify calibration
provenance, the native caller's upper-frequency masking, parameter-space
coverage, or physical accuracy against NR/hybrids.

The [IMRPhenomXAS paper, Sec. V, Eqs. (64)–(65)](https://arxiv.org/html/2001.11412v1)
states that the public LAL implementation uses an arbitrary upper cutoff at
$Mf=0.3$. The native carrier intentionally continues to evaluate above that
limit. The test requests 20–512 Hz at $\Delta f=0.125$ Hz and compares every
sample with $Mf<0.3$, without amplitude rescaling, phase or time adjustment,
or sample selection based on residuals. It also asserts that the public binary
is exactly zero at all requested bins beyond the published limit and nonzero
at every bin inside it. For the 120 $M_\odot$ negative-spin holdout, this
boundary is 507.5636 Hz: the final compared bin is 507.5 Hz and the first
zero binary bin is 507.625 Hz. If one compares the native uncut tail against
those wrapper zeros, the raw full-grid relative complex $L_2$ is
$9.49\times10^{-9}$ (local diagnostic), entirely from those 35 bins; this
is a wrapper-contract difference, not a discrepancy in the nonzero carrier.

The frozen cases come from the [oracle catalog and holdout builder](https://github.com/xangma/diffgw/blob/main/tests/parity_harness.py):

| Case | $m_1,m_2$ ($M_\odot$) | $\chi_1,\chi_2$ | $f_{\rm ref}$ (Hz) | Inclination | $\phi_c$ | Distance (Mpc) | Last compared bin (Hz) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Catalog | 35, 30 | 0.2, -0.3 | 20 | 0.6 | 0.4 | 500 | 511.875 |
| Unequal positive | 45, 20 | 0.35, 0.6 | 35 | 0.9 | 0.71 | 680 | 511.875 |
| Unequal negative | 90, 30 | -0.4, -0.6 | 27 | 1.2 | 0.71 | 680 | 507.5 |

| Backend | Catalog $h_+,h_\times$ | Unequal positive $h_+,h_\times$ | Unequal negative $h_+,h_\times$ |
| --- | --- | --- | --- |
| Torch | 5.2594571e-14, 5.25902966e-14 | 3.87238823e-14, 3.87206571e-14 | 1.87198555e-13, 1.87215954e-13 |
| JAX | 4.52875487e-14, 4.52837366e-14 | 4.1805268e-14, 4.18020341e-14 | 1.88433315e-13, 1.88455793e-13 |

Only the public `SimInspiralChooseFDWaveformSequence` binary API was called.
No LAL implementation source was inspected or translated, and no coefficient
or threshold was fitted to the binary. The fixed physical mass conversion
uses the [independently audited SI/IAU constants](../math/units.md).

On `len`, in `~/repos/diffgw-worktrees/xas-whole-waveform-gate`, run:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu DIFFGW_REQUIRE_LAL_BINARY=1 \
  .venv/bin/python -m pytest -q -s tests/test_imrphenomxas_lal_binary.py
```

The fresh Python 3.12.3 virtual environment used NumPy 2.5.3, SciPy 1.18.1,
SymPy 1.14.0, Torch 2.14.0+cpu, JAX/JAXLIB 0.11.2, pytest 9.1.1, and the
CI-pinned LALSuite 7.26.15. The test is also included in the pinned-binary
CI job with missing oracle/backend dependencies treated as failures there.
The focused run of this binary test together with
`tests/test_imrphenomxas_waveform_math_reference.py` passed **99/99** on `len`.
