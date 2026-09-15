# Independent event and attachment review

Date: 2026-09-12. Read-only review of `eob/events.py`,
`EOBTrajectoryIntegrator.integrate_to_frequency_peak`, `eob/attachment.py`,
and their two focused test files. The existing NQC and merger modules were
read to verify their interface and matching algebra. This report is the
only repository change made by this review. No target implementation,
binary oracle, NR data, or external source port was accessed.

## Findings

No actionable implementation defect was found in the reviewed event,
five-condition NQC solve, or complex C1 attachment paths. This conclusion
does not validate a complete physical SEOBNRv4 trajectory or public driver.

There is an integration coverage requirement for the driver:
`integrator.py:127` and `integrator.py:176` stop each trajectory after two
samples beyond its observed orbital-frequency maximum. The published
NQC offset can be positive: `nu=.001, chi=-.5` gives `+1.04387817688803 M`,
and `nu=.01, chi=-.4` gives `+.217696068195 M`. Consequently these retained
samples need not cover the later matching event and its five-point stencil.
`attachment.py:95`–`attachment.py:97` correctly reject insufficient valid
history. A combined driver must retain enough additional post-peak history
for this stencil or explicitly fail for the unsupported coverage. This is
not a silent attachment error; the bounded event helper documents its
two-sample stopping rule. The primary agent was informed before driver
integration was completed.

## Reviewed reasoning

Completed rows have zero stage derivatives before later RK stages are
formed (`integrator.py:151`, `integrator.py:180`). Their accepted states
remain frozen, while the mask retains the last actually evolved sample.
The first-maximum stopping rule is explicitly limited to smooth inspiral
and plunge. Hamiltonian failures, nonfinite stages, nonpositive orbital
frequency, and exhausted steps raise rather than creating an endpoint peak.

The peak interpolation uses centered, scaled times and removes the central
frequency before solving. It brackets the derivative root, safeguards the
Newton step, and requires negative curvature. Gradients follow a fixed
selected stencil; the discrete maximum selection is outside that guarantee.

The attachment polynomial contains `A`, three products `A*q_i`, two phase
bases `p_i`, and the supplied positive base frequency. Therefore its local
amplitude is `A+sum(a_i*A*q_i)` and its local phase is an integral of the
base frequency minus `sum(b_i*p_i)`. These are exactly the combinations
whose first two amplitude derivatives and first frequency derivative enter
the five NQC conditions. Anchoring to the first base-mode phase retains
the complex NQC correction's sign. The merger uses the same phase and
target amplitude, zero amplitude slope, and target frequency, establishing
the complex C1 join mathematically. The earlier linear interpolation has
its separately documented numerical behavior.

Input mode amplitudes have the fixed `(D/M)*h22` normalization. Arbitrarily
rescaling the base mode while leaving the published targets fixed is a
different calibration problem; the complete output should not be expected
to have a free amplitude-scale covariance under that change.

## Independent checks

The two focused test files pass: **14 passed**. They include the five
conditions evaluated through output derivatives, complex C1 matching,
time-origin and global-phase covariance, peak-location gradients, and
independent batch stopping. Additional read-only probes established:

- A combined gradcheck through `nu`, `chi`, QNM frequency/damping, global
  phase, and base-amplitude normalization passes.
- Two independently terminated, padded trajectories with different peak
  times, intrinsic parameters, and QNMs produce exactly the same output as
  separately truncated scalar calls; both maximum absolute errors are zero.
- A nonuniform-time quartic peak passes gradcheck and lies within the
  immediate-neighbor bracket. Under `t -> 3*t+17` and `Omega -> 7*Omega+11`,
  peak-time and curvature covariance errors are approximately `1.1e-14`
  and `7.9e-15` respectively.
- Float32 attachments remain finite for spacings from `.4 M` to `.005 M`.
  Sampled left/right derivative differences at a `1e-4 M` separation are
  roughly `3e-5` to `1e-4` relative in tested cases. Tight numerical matching
  should use float64 and check spacing convergence; exact C1 is an
  algebraic statement subject to floating-point conditioning.

These checks do not measure NR agreement, phase-inheritance correctness,
remnant-block selection, or actual-orbit convergence. No full-suite or
public-driver verification is claimed here.

## Reviewed snapshot

| File | SHA-256 |
|---|---|
| `events.py` | `2fcb3dc9e7362fd8c15d8f78d3ddcb7b603d5d03dbdebb188eca460e09dd4f75` |
| `integrator.py` | `125b9d86521e05f806b16dd5842fa4da68c76038fd20eb669d0e7890feb21c55` |
| `attachment.py` | `09941c1cc1aa371898205a38719f13d359655f0b9cad76828ea86cfb09df0521` |
