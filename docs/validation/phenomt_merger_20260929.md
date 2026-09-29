# IMRPhenomT-family dominant merger qualification receipt — 2026-09-29

## Scope and inputs

The versioned [paper-derived scalar source](../math/phenomt-merger.md)
reconstructs the updated THM Eqs. (28)–(32) dominant merger frequency,
slope, phase difference, and signed amplitude from **supplied** cutoff,
interior, peak, and QNM inputs. It imports neither native backend nor fitted
target functions. This is a component equation check across T/THM and the
shared aligned carrier used by TP/TPHM, not a whole-waveform qualification.

Source/test SHA-256, identical in local worktree and len snapshot:

| File | SHA-256 |
| --- | --- |
| `src/diffgw_reference/phenomt_merger.py` | `36d2518cdf45865faf87935b093ef9c891f49df8da620abe982a8e746a575117` |
| `tests/test_phenomt_merger_math_reference.py` | `a16faa1913849882c7e9380edff823bf7f8828a5e9a78d7657321f41fa62e89c` |

## Execution

Host `len`, directory `/home/xangma/tmp/diffgw-phenomt-qualification-20260929`,
Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2 in FP64, `mpmath` 1.3.0,
SymPy 1.14.0, LALSimulation 6.2.1. The imported `_lalsimulation` extension's SHA-256 is
`e816879d23e652e8b3928496d6fd261de0832b5c3781c50e33a458042495b42e`.
The frozen tests compare equal masses, unequal aligned spins, and a 10:1
mass-ratio case for each backend.

| Check | Command suffix after `PYTHONPATH=src:. python -m pytest -q` | Observed result |
| --- | --- | --- |
| New merger plus existing inspiral/ringdown scalar gates | `tests/test_phenomt_merger_math_reference.py tests/test_phenomt_ringdown_frequency_math_reference.py tests/test_phenomt_ringdown_amplitude_math_reference.py tests/test_phenomt_inspiral_collocation_math_reference.py` | 29 passed in 67.66 s; `phenomt_scalar_combined.log` |
| Aligned THM and NNLO TP/TPHM against pinned binary | `tests/test_phenomt_family_lal.py::test_lal_binary_reference` | 9 passed in 11.87 s; `phenomt_aligned_nnlo_binary.log` |
| q=10 MSA/PV300 strict binary gate | `-rx tests/test_phenomt_family_lal.py::test_high_mass_ratio_precession_unresolved_gate` | 4 strict expected failures in 31.64 s; `phenomt_q10_gate.log` |

The merger tests require coefficient, frequency, slope, and amplitude
agreement at `rtol=2e-10`, `atol=2e-11`, and phase differences at
`rtol=2e-10`, `atol=2e-9`. They compare off-collocation times as well as
boundary values. Scalar self-checks use a SymPy expression, 75-digit
collocation, and symmetric derivative probes. Both Torch and JAX pass all
three cases.

## Interpretation and remaining gates

The passing result checks the published interpolation algebra and the native
join choices when fitted values, PN cutoff data, and QNM rates are supplied.
The THM fit surfaces and raw NR/EOB targets are not recreated by the paper;
the pinned MIT `phenomxpy` asset remains their identified implementation
source. Mode-specific HM merger, signed zero crossings, remnant/QNM lineage,
precessing ringdown input splice, and complete waveform accuracy remain
separate gates.

All four q=10 MSA/numerical TP/TPHM gates still fail their retained binary
thresholds. The [phase and rotation blocker](phenomt_q10_gate_blocker_20260927.md)
localizes the discrepancy but does not establish a paper-backed production
correction. This receipt changes no precession equation, option, or threshold.
The prior measured q=10 TP/TPHM relative L2 values are `0.0202333/0.0225165`
for MSA (limit `0.01`) and `0.308418/0.307103` for numerical (limit `0.03`).
The published-equation MSA scalar matches native before the binary's
pre-MECO rotation diverges; a binary-compatibility spin-contraction choice
reduces that residual but lacks paper provenance. The independently integrated
numerical trajectory matches native within `4.78e-9` rad while the public
binary's reconstructed rotation differs by up to `0.39276` rad. These
observations locate the remaining work in option/convention and pre-merger
orientation qualification, not in the newly checked merger interpolation.
