# TEOB ringdown and sampling assembly clean-room verification

This component implements the separate **Ringdown and sampling assembly**
section of the [TEOB merger assembly contract](../../docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md).
It consumes corrected full and auxiliary arrays from the explicit NQC hand-off
(`time`, `amplitude`, `phase`, and auxiliary orbital frequency), joins the
charts, finds the auxiliary frequency peak, appends the fixed 1500-knot
support, and attaches the existing clean `postmerger_22` ansatz.  It does not
infer NQC targets, evolve a trajectory, or alter production dispatch.

The public entry point is
`assemble_teob_ringdown_22(full_time, full_amplitude, full_phase,
auxiliary_time, auxiliary_amplitude, auxiliary_phase,
auxiliary_orbital_frequency, nu, final_mass, final_spin, ...)` in
`src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py`.  The
`CorrectedWaveformSeries` record names the three corrected waveform arrays
exchanged by the NQC stage.  `RingdownAssemblyResult` returns the final mode,
joined and extended signed amplitude/phase charts, the ansatz chart, event
indices/times, attachment times in both time units, and opaque matching
metadata.  Its `epoch`/`coordinate_start` is the first coordinate of the
returned support; it is not a peak-shifted observer epoch.

The implementation follows the contract's strict join (full samples before
the auxiliary first knot, then auxiliary samples from its second knot), the
seven-knot degree-six peak polynomial and one Newton step, and the ordered
threshold `t_peak/final_mass - 3/final_mass + 2/final_mass`.  The last
strictly earlier joined knot and its predecessor are returned explicitly.
Amplitude and phase remain separate through the join and are interpolated
separately on an optional output grid, so a signed corrected amplitude is
preserved before the postmerger ansatz replaces the tail.  Natural cubic
interpolation uses the independently derived tridiagonal natural-endpoint
system with a Thomas solve; this keeps physical O(10^4)-knot supports linear
in storage rather than constructing a dense quadratic-size matrix.

Run the focused checks with:

```text
PYTHONPATH=src:tests:scripts:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_teob_ringdown_assembly_clean.py
```

The tests independently reconstruct the peak polynomial and phase anchor,
check exact and off-knot attachment arithmetic (including a floating-point
boundary), support and extension counts, time translation, signed amplitude
and no-rescaling behavior, natural-cubic interpolation against SciPy on a
nonuniform knot grid, asymptotic QNM behavior, batches/gradients, and invalid
domains.  The suite is synthetic and bounded.

Public fixture inspection gives potential chart sizes of 19,359 joined and
20,859 extended knots for `torchwave-public-eob-dt_0p5M.npz`, and 14,657
joined and 16,157 extended knots for `torchwave-public-eob-dt_1_4096s.npz`,
using the fixture state times solely to count the contract join.  No corrected
NQC arrays were available for this freeze, so no physical assembly or full
trajectory validation is claimed.  The report records hashes and the clean
natural-cubic timing on those support sizes.
