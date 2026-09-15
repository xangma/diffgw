# Clean-room pass 24: TEOBResumS multipolar flux audit

## Finding

The current TEOBResumS path uses the deliberately labelled prototype
`EOBRadiationReaction`: a positive Newtonian flux with 1PN and 1.5PN
corrections. The primary TEOBResumS publication instead defines the azimuthal
reaction from the sum of factorized multipolar waveform fluxes. Its mode has
the structure

\[
h_{\ell m}=h_{\ell m}^{(N,\epsilon)}\hat S_{\rm eff}^{(\epsilon)}
\hat h^{\rm tail}_{\ell m} f_{\ell m}\hat h^{\rm NQC}_{\ell m},
\]

with the NQC factor entering the flux during the self-consistent construction.
The 2018 paper later removes NQC corrections from the flux in its final
practical choice, but still requires the complete TEOBResumS residual mode
set and its model-specific source factors.

## Why no flux replacement was made

TorchWave already contains an independently reconstructed DIN circular asset in
`FactorizedModes`: the nonspinning residual table covers the published DIN
35-mode set through \(\ell=8\), and the optional Pan asset supplies only
leading finite-mass spin terms through \(\ell=4\). It is explicitly documented
as an independent factorized building block, not the calibrated TEOBResumS
flux.

A source-complete general TEOBResumS replacement is therefore blocked by
specified inputs that are absent from this repository:

1. the full aligned-spin residual-amplitude series for every mode retained by
   the TEOBResumS flux (the available spin asset is only leading order and
   stops at \(\ell=4\));
2. the TEOBResumS tidal multipolar corrections needed when either tidal
   deformability is nonzero; and
3. the exact final model mode-selection and any associated updated residual
   factors from the later TEOBResumS references. The 2018 paper identifies the
   corrected odd-\(m\), \(\ell=5\) factor, but does not provide a complete
   standalone replacement table for every spinning/tidal mode in the paper
   itself.

Switching to the available DIN asset would silently change the model for
spinning or tidal inputs and would not be a source-complete TEOBResumS
correction. The prototype remains in the TEOBResumS path until those inputs are
independently specified.

## Independent checks

`test_teob_prototype_flux_has_newtonian_weak_field_limit` checks the published
quadrupole limit
\(\mathcal F\to(32/5)\nu^2\Omega^{10/3}\) at two weak-field frequencies.
The preceding TEOB radiation test checks
\(\nu\Omega F_\varphi=-\mathcal F\) and the independently justified
TEOBResumS choice \(F_{r_*}=0\). The existing factorized-mode tests separately
check the DIN multipolar flux asset and its mechanical-power identity.

## Frozen public binary measurement

No production flux code changed in this audit, so the strict before/after
measurement is intentionally identical. The fixture is `m1=m2=1.4`, zero
spins/tides, `f_lower=30 Hz`, `f_final=500 Hz`, a 0.5 Hz grid, and one Torch
thread; the comparison uses only the public `lalsimulation.SimInspiralFD`
binary output:

| polarization | strict complex L2 before | strict complex L2 after |
|---|---:|---:|
| plus | 1.0009161561301 | 1.0009161561301 |
| cross | 0.9993295253941 | 0.9993295253941 |

The audit therefore records a justified implementation boundary rather than an
oracle-derived flux substitution.
