# Pass 26: SEOBNRv4 calibrated inputs and initial data

Date: 2026-09-13

This pass audited the earliest quantities that can affect the phase of the
sufficient-horizon `SEOBNRv4Published` path.  The implementation inputs were
limited to the primary SEOBNRv4 paper
([Bohé et al., arXiv:1611.03703](https://arxiv.org/abs/1611.03703)) and the
primary quasi-circular initial-data algorithm
([Buonanno, Chen & Damour, arXiv:gr-qc/0508067](https://arxiv.org/abs/gr-qc/0508067)).
Public binary output remains a validation oracle only.

## Published-input inventory

| Dependency | Published specification | Current path |
| --- | --- | --- |
| Conservative calibration | The v4 paper prints the global (K,d_{SO},d_{SS}) polynomial fits in Sec. 4. | `SEOBNRv4Hamiltonian.calibration_parameters` and the radial potentials transcribe those equations. |
| Circular initial data | Buonanno et al. Eqs. (70)--(71) solve the spherical orbit; Eqs. (72), (76), and (77) add the leading radiation-reaction radial velocity and convert it to (p_{r*}). | `find_quasicircular_initial_conditions` solves the same two circular equations and applies the Eq. (76)--(77) mapping. |
| Radiation reaction | v4 Eq. (2.9) specifies the torque from the summed mode flux and excludes NQC from that flux. | `SEOBNRv4PublishedFlux` and its torque adapter implement this equation. |
| NQC targets and time shift | Appendix A and Sec. 4 print all target fits and the calibrated time offset. | `nqc.py` implements the printed targets and offset. |
| Merger-ringdown coefficients | Appendix B prints the (c_i,d_i) fits. | `merger_ringdown.py` implements the printed formulas. |
| Attachment interpolation | The paper supplies matching conditions and time shifts, but no unique numerical interpolation stencil. | `attachment.py` uses a local quartic interpolation; this is an implementation choice, not a missing calibrated table. |

The v4 calibration coefficients are therefore not missing inputs.  The raw
NR/Teukolsky training samples and MCMC metadata are not supplied as a
machine-readable initial-condition or trajectory table in the primary paper
or its public arXiv source.  Reconstructing such a table from waveform
residuals would be an oracle fit and is outside this clean-room pass.

## Initial-data equation audit

For a spherical circular branch (H_{,r}=0), the published algorithm gives

\[
 j'(r)=-H_{,rr}/H_{,rj},\qquad \dot r=F_\phi/j'(r),
\]

and, at (p_{r*}\to0),

\[
 \dot r=\left[\xi H_{,p_{r*}p_{r*}}/H_{\rm real}\right]p_{r*}.
\]

The current implementation consequently uses

\[
 p_{r*}=F_\phi\,\frac{H_{\rm real}}
 {\xi H_{,p_{r*}p_{r*}}j'(r)},
\]

which is the leading-radiation-reaction prescription in Eqs. (72), (76), and
(77), rather than an empirical v4 fit.  The new invariant test recomputes
the derivatives from the returned SEOBNRv4 state and verifies the torque
balance for three independent mass-ratio/spin/frequency points.

Command and result:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/test_clean_room_eob_conservative.py
13 passed
```

## Decision and metric

No equation-level correction is justified.  The earliest phase-sensitive
quantity that looked approximate, (p_{r*}), is already the published
leading-order initial-data construction; the papers provide no alternative
calibrated value to substitute.  The strict frozen diagnostic is unchanged:
relative L2 `1.0206987416256423` for polarizations and
`1.0205836988469421` for the projection.  These are both before and after
values because this pass changes only the independent invariant test and
documentation.

The remaining parity gap must therefore be attributed to an unresolved
physics or numerical choice (including the paper-unspecified attachment
interpolation) rather than to a source-complete missing v4 calibration input.
