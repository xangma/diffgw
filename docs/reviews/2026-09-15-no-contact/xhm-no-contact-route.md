# XHM no-contact data route

Date: 2026-09-15. Scope: bounded metadata check for a public, independently
reachable calibration artifact. No LAL/PyCBC/TEOB implementation source,
inherited fit body, waveform output, or private cache was read.

## Finding

The current `122022` target still has no admissible public numerical bundle in
the bounded records listed below. A clearly versioned *older* paper target is
reachable as the arXiv `2001.10914v2` ancillary source archive:

* URL: <https://arxiv.org/src/2001.10914v2>
* observed SHA-256: `cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`
* metadata inventory: 8 mixing-coefficient files, 10 QNM files, 37 amplitude-fit
  files, 33 phase-fit files, and the two fit notebooks plus `README`.

The archive README identifies the fit and notebook contents as paper-era
assets. The arXiv record links the article to its
[non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/),
which grants arXiv distribution of the article and does not establish an MIT
redistribution grant for the ancillary fit bodies. The public location alone
does not clear use or distribution: this is a metadata-verified candidate,
not an approved calibration source. This review does not authorize bundling
it into torchwave or relabelling it as release `122022`; numerical correctness and
calibration provenance remain separate gates.

## Checked records and remaining candidate

* [arXiv paper record, v2](https://arxiv.org/abs/2001.10914v2), revised
  2020-11-03: paper equations and ancillary-file identity (paper-era only).
* [arXiv source archive, v2](https://arxiv.org/src/2001.10914v2): member
  inventory and observed archive digest above; no `122022` release marker.
* [arXiv license record](https://arxiv.org/licenses/nonexclusive-distrib/1.0/):
  non-exclusive article distribution terms; no MIT ancillary-data grant.
* [LALSimulation waveform-parameter documentation](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_inspiral_waveform_params_8c_source.html):
  a setter/version metadata candidate referenced by prior audit records.
  This page was not fetched or inspected in this pass and supplies no new evidence.
* Local release-control record
  `docs/reviews/2026-09-13-parity-review/verification/xhm-inspiral/xhm-inspiral-public.json`
  (SHA-256 `2ea0c7d296dc9b3138cccbdce0433770bf940f71e197a1d031ce0928449329e1`):
  observed default `122022`, amplitude-fit `122018`, variant `3`.

An independently calibrated model could plausibly reach useful scientific
fidelity from the published equations and independently chosen calibration
data with suitable terms, with a new model/version and validation set. Such
recalibration alone would not establish exact LAL `122022` parity.

## Executable check

`scripts/xhm_no_contact_archive_check.py` verifies the archive hash and member
inventory without extracting or evaluating any calibration body. For the
observed archive, run (the digest argument is required for identity
verification):

```text
python scripts/xhm_no_contact_archive_check.py /path/to/2001.10914v2.tar.gz \
  --sha256 cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e
```

With no `--sha256`, the report marks `sha256_status` as `unchecked`, sets
`inventory_verified` independently, and keeps `identity_verified` and overall
`verified` false. The checker is deliberately metadata-only; it does not
supply fit values.

The coordinator independently reproduced the archive hash and all counts:
[verification record](verification/xhm-archive.json).
