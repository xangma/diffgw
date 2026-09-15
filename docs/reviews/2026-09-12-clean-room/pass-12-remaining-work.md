# Remaining parity work after the pass-12 scope review

This is a parent assessment, separate from the implementers' source derivations.
The previous frozen result is 11/21 catalog cases and 4/13 holdouts. Those are
case counts, not a percentage of the implementation or a parameter-domain
certificate. Ten catalog cases and nine holdouts remain unsuccessful; one
catalog case lacks a complete independent reference. No defensible completion
time can be inferred from test counts.

## Immediate implementation priorities

1. Finish the current source-supported XHM correction and measure it on the
   unchanged cases. Remaining phase-node targets, absolute phase choices and
   spheroidal 32 conventions need explicit definitions tied to the existing
   calibration tables. Do not replace them with the newer amplitude ansatz
   unless the corresponding fit data are available and identified.
2. Complete the MSA precession/remnant combination before presenting XP/XPHM
   as compatible with the documented default model. Pratten et al.
   arXiv:2004.06503v2, Table III and the numerical-relativity injection section,
   specify 223/FinalSpinMod3/convention1 defaults. Native default NNLO102 is a
   distinct prescription. The initial/final mass inconsistency in Eqs. 4.18
   and 4.29 remains a source question; simply selecting the printed initial-M
   denominator breaks the usual aligned Kerr-spin limit.
3. Resolve remaining PhenomD/NSBH phase and calibration details and detector
   geometry precision using independent source information. Two PhenomD
   holdouts meet the polarization L2 bound but fail the detector projection
   bound; changing acceptance thresholds would not resolve that discrepancy.

The parent public-binary-only control in
`verification/pass-12/public-defaults.json` finds bitwise equality between
reference defaults and explicit 223/3 for the four fixed XP/XPHM cases.
Explicit 102/0 and 320/2 differ. This establishes finite-case output
comparability, not the reference's internal dispatch, and does not determine
implementation coefficients or resolve the MSA mass convention. Acceptance
cases and thresholds remain untouched.

Colleoni et al., [arXiv:2412.16721v1](https://arxiv.org/html/2412.16721v1),
Sec. II, describes updated higher-mode amplitudes and SpinTaylorT4 precession.
The paper explicitly selects 320/2 for its numerical-precession model.
It is a distinct available prescription, not evidence that the fixed reference
cases use it. The newer amplitude formulas require their corresponding
calibration fits. Existing 2020 tables cannot automatically be substituted.

## Larger unfinished model implementations

- SEOBNR_ODE: connect a complete, calibrated SEOBNRv4 dynamics/radiation model,
  resolve the published driver choices, and validate NQC and merger attachment
  against independent complete waveforms. The current default remains a
  prototype even though individual published components are implemented.
- SEOBNR_ROM: follows a validated generator. Its existing surrogate represents
  TorchWave's earlier generator; train and independently validate new weights
  after completing the target model.
- TEOBResumS: a separate dynamics, radiation, tidal and merger implementation.
  The native source still labels important terms and attachment choices as
  prototype heuristics.
- KerrRingdown and TaylorF2_HM: complete missing waveform reference/convention
  coverage. A carrier or eigenfrequency check does not validate higher modes
  or an entire ringdown signal.

## Work policy for subsequent passes

Judge progress by completed waveform behavior and unchanged external gates.
Tests establish correctness of a specific implementation change but do not
count as additional parity successes. Keep Luna assignments bounded to one
source-supported change; reject or explicitly record unresolved source gaps
instead of repeatedly adding helpers around them. Finish one model milestone
before broadening implementation scope. Existing tables remain in use and the
user-deferred NR simulation is not required to resume this work.
