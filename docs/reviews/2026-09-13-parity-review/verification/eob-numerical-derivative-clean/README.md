# Clean numerical derivative verification

This directory records a bounded, opt-in Torch experiment.  The implementation treats the existing Hamiltonian as a callable API and leaves the production analytic derivative path and driver defaults unchanged.

The derivative uses the contract orientation `(position, momentum) = ((r, 0, 0), (p_r*, j/r, 0))`.  It evaluates the full real Hamiltonian divided by `nu`, applies the four-point central rule at `x-h`, `x-h/2`, `x+h/2`, and `x+h`, and uses the two-point rule as a quadratic truncation check.  The local spacing update models the measured error as `a/h + b h^2`, caps each update to one decade, and selects the lower estimated error row by row.  The spacing and error diagnostics are detached selection data, while the selected function values retain ordinary Torch gradients.

At exactly `nu=0`, the value `H_real/nu` is replaced by its finite coordinate-derivative limit, the effective Hamiltonian.  The conservative field is then assembled from the Cartesian partials:

```text
dr/dt       = xi H_px
Omega       = H_py / r
dp_r*/dt    = xi (-H_x + H_py j/r^2)
```

The public central-difference convention is documented in the [GSL numerical differentiation manual](https://www.gnu.org/software/gsl/doc/html/diff.html).  GSL is not a runtime dependency of this clean experiment, and no external implementation is imported by the diagnostic.

Run the bounded report from the repository root with:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python scripts/diagnose_eob_numerical_derivative_clean.py --output docs/reviews/2026-09-13-parity-review/verification/eob-numerical-derivative-clean/pointwise-report.json
```

The generated [pointwise report](pointwise-report.json) covers quartic-polynomial and sine derivatives plus two short state batches.  With `initial_step=1e-4` and one update, each of the three partials used eight Hamiltonian evaluations per state (48 total per two-row case).  The maximum absolute difference from the clean analytic conservative path was `9.3e-12` for the nonspinning case and `4.1e-11` for the aligned-spin case.  The analytic function checks were below `4.0e-11` for the quartic polynomial and `7.3e-12` for adaptive sine in float64.

These are pointwise floating-point checks, not a waveform or inspiral validation.  The adaptive choices are independently derived diagnostics; undocumented details of any native implementation, including its exact acceptance logic or iteration policy, are not inferred here.

The follow-up fixture section of the report attributes the effect of the
numerical orbital frequency on the attachment.  For each public post-peak
fixture, the analytic and numerical frequency series are passed separately to
`attach_seobnrv4_22_numerical_compatibility` with the same published carrier
`base_mode`/`base_frequency`, remnant, target parameters, and full absolute
output grid.  The choices are
`SEOBNRv4Hamiltonian(nu=0.24, chiS=0, chiA=0, delta=0.2)`,
`SEOBNRv4PublishedFlux(lmax=8, spin_truncation="generic-leading-ell8",
retain_pan_cubic_spin=True)`, `spin_v6="bohe_total_energy"`,
`include_pan_v8=False`, `include_positive_chi_540=False`, and
`retain_pan_cubic_spin=True`.  The final spin uses Hofmann fit `(3, 4)` and
the signed QNM-table convention.  The mode scale is evaluated as
`50*lal.MTSUN_SI*lal.C_SI/(100e6*lal.PC_SI)` without importing LAL.

The analytic attachment reproduces the clean component baselines: on the
`dt_0p5M` fixture, post-boundary raw and amplitude relative L2 are
`4.4229410578e-5` and `2.7970093675e-7`; on `dt_1_4096s` they are
`3.2103532724e-6` and `2.7954724314e-7`.  Replacing only its orbital
frequency with the bounded finite-difference series changes the raw L2 to
`8.1616261843e-5` and `1.6331670744e-4`, respectively, while amplitude L2
changes by less than `6e-13`.  The peak time, match time, boundary time, and
boundary sample index remain unchanged on both grids.  The match phase shifts
by `3.7394e-5` rad and `1.6643e-4` rad; the boundary frequency shifts by
`-4.8112e-10` and `1.4291e-8`.  The largest coefficient component shifts are
`2.5630e-3` and `1.1372e-2`, in that fixture order.  These residuals are
component-attribution measurements and do not select a derivative policy.

The public fixture hashes are:

| fixture | SHA256 | finite-difference Hamiltonian evaluations |
| --- | --- | ---: |
| `torchwave-public-eob-dt_0p5M.npz` | `7bf76ba9b81515bdd22377a8569bc0ee8809717baeb43d7f887b71ecdeb9d88a` | 231696 |
| `torchwave-public-eob-dt_1_4096s.npz` | `eefe5b8f1380601f82232e78b0e3a8e1b8abf5a4f9cfb5208e28331b8f5685f9` | 234312 |

The report also records SHA256 values for the clean diagnostic, derivative,
attachment, waveform, flux, Hamiltonian, remnant, event, NQC, and QNM helper
sources.  This remains a bounded post-peak batch diagnostic; no full
trajectory or inspiral validation was run.
