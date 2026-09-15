# Pass 21: EOB integration

Pass 21 connects the existing clean-room SEOBNRv4 equation-level components
to the waveform API. The prototype `SEOBNR_ODE` path now attempts refined peak
detection followed by the native five-condition NQC solve and C1 merger/
ringdown attachment when the trajectory resolves a usable peak. Under-resolved
runs retain the legacy final-sample PhenomD damped-exponential fallback.

The explicit `assembly_model='seobnrv4_published'` option exposes the existing
end-to-end published assembly through `SEOBNR_ODE`. It requires explicit values
for all five unresolved choices: spin truncation, the v6 spin-phase convention,
the PAN v8 switch, the positive-chi 5.40 switch, and final-spin fit orders.
No choice is presented as LAL-compatible. The pass also adds the Hamiltonian
phase/radiation adapter and positive-domain validity guards for orbital
frequency and angular momentum.

Validation evidence:

- The focused pass-21 tests passed 28 tests in 14.01 seconds.
- The combined focused EOB suite passed 205 tests with 4 skipped in 24.05
  seconds.
- The complete ordinary suite passed 1,161 tests with 9 skipped across four
  parallel shards; the slowest shard finished in 50.28 seconds.
- The post-change strict public-binary diagnostic still fails. Its maximum
  complex relative L2 errors are 1.0206987416256423 for polarizations and
  1.0205836988469421 for the detector projection.

The strict result is unchanged and remains far outside the `1e-6` acceptance
limit. This integration therefore improves the executable model path and its
component coverage but does not establish LAL parity. The residual phase
convention, Hofmann-block selection, time/phase semantics, and numerical parity
remain unresolved.

The implementation uses primary papers and independently published data; no
LAL or PyCBC implementation source was used as implementation input.
