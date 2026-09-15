# Pass 32: clean-room PhenomD asset search

This pass searched for independently released, source-eligible assets that
could resolve the remaining small IMRPhenomD residuals. The search was limited
to the primary Khan/Husa papers and source archives, author-hosted mathematical
data, and public data pages. No LAL, PyCBC, TEOBResumS, translated
implementation, or implementation-derived table was inspected or used.

## Sources searched

- Khan et al., *Frequency-domain gravitational-wave model of nonprecessing
  black-hole binaries*, [arXiv record](https://arxiv.org/abs/1508.07253v2),
  [source archive](https://arxiv.org/src/1508.07253v2), and
  [published article](https://doi.org/10.1103/PhysRevD.93.044007).
- Husa et al., *Frequency-domain gravitational waves from nonprecessing
  black-hole binaries. I.*, [arXiv record](https://arxiv.org/abs/1508.07250v2),
  [source archive](https://arxiv.org/src/1508.07250v2), and
  [published article](https://doi.org/10.1103/PhysRevD.93.044006).
- E. Berti's [author QNM data page](https://pages.jh.edu/eberti2/ringdown/)
  and its linked [Kerr fit table](https://pages.jh.edu/eberti2/ringdown/fitcoeffsWEB.dat).

The Khan source archive contains paper source, bibliography, and PDF material;
the Husa archive contains paper source, bibliography, PDF, and figure material.
Neither contains a machine-readable `.m`, `.mat`, `.dat`, `.csv`, `.json`, or
notebook asset containing the full-precision PhenomD calibration vectors or the
QNM interpolation knots. The archive hashes and file inventory are recorded in
[Pass 18](pass-18-phenomd-assets-primary.md).

The published Khan paper gives the fit map and rounded Table V values. It does
not expose the extra binary digits used by the native unrounded vectors. Husa
states that dimensionless `M omega_RD` is obtained by interpolating Berti data,
but does not publish the interpolation knots, order, endpoint rule, or a
machine-readable table. The APS and arXiv pages inspected during this pass did
not expose a public PhenomD coefficient supplement or data endpoint.

## Independent Berti asset

Berti's author page says that the QNM data and routines are freely available
with citation requested. It links `fitcoeffsWEB.dat`, whose documented columns
are `(l,m,n,f1,f2,f3,q1,q2,q3)` for

```text
M omega_R = f1 + f2 (1-a/M)^f3
Q         = q1 + q2 (1-a/M)^q3.
```

The browser-rendered table is an independent author asset, but its displayed
fit coefficients are rounded (for example the 220 row is
`1.5251,-1.1568,0.1292,0.7000,1.4187,-0.4990`). Those values already match the
existing `BERTI_QNM_FITS` entries in
`src/torchwave/approximants/kerr_ringdown.py` for the supported modes. The
asset therefore supplies neither the missing full-precision PhenomD Table V
vectors nor additional interpolation metadata sufficient to replace the
current independent `evaluate_kerr_qnm_220` path.

The host returned HTTP 403 for repeatable command-line retrieval, including a
normal browser user-agent and referer, so no byte hash is recorded. The
author-page statement is a request for free use with citation, not an explicit
software or data license. The table was not copied into the repository.

## Decision

No exact clean-room asset was found that justifies a runtime correction. The
Berti table is already represented in the dedicated Berti backend, and replacing
the IMRPhenomD evaluator with its rounded coefficients would be an unsupported
guess. No fit to oracle residuals was performed. No runtime or test file was
changed in this pass; no focused parity run is warranted for an unmade change.

The remaining evidence-backed blocker is an admissible author release containing
the unrounded PhenomD calibration vectors and/or the precise Berti interpolation
data and algorithm used by the original model.
