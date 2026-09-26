# TaylorF2_HM independent-oracle search — 2026-09-26

## Outcome

No qualifying published numerical oracle was located. No external-parity test
or numerical claim was added. The existing author-source, paper-fidelity,
scalar-reference, Torch/JAX-parity, angular, and dominant-mode gates retain
their narrower labels.

This was a bounded clean-room search. A same-equation SymPy evaluation, a
translation of the Mishra supplement, a LAL-derived series, or a fit to diffGW
output was rejected as independent evidence.

## Required oracle record

A closing record must supply all of the following without inference or fitted
convention alignment:

- complex frequency-domain values, or per-mode $(h_+,h_\times)$ plus their
  exact sum, for `(2,2)`, `(2,1)`, `(3,1)`, `(3,2)`, `(3,3)`, `(4,2)`,
  `(4,3)`, and `(4,4)`;
- component masses, nonzero aligned spins, distance, inclination, phase and
  time origins, frequency grid and units;
- a finite `f_ref`, its phase definition, and the Fourier, tetrad,
  spin-weighted-harmonic, and signed-mode conventions;
- the termination rule for every mode, with samples below, at, and above each
  per-mode cutoff;
- an author-published table or independently maintained executable, versioned
  provenance and hashes, and explicit redistribution terms.

The comparison must be direct, complex, per mode and on the sum in both Torch
and JAX. A dominant carrier, angular basis, magnitude-only ratio, or post-hoc
phase/time alignment cannot close the gate.

## Sources examined

| Source | Published material found | Missing for this gate |
| --- | --- | --- |
| [Mishra et al., arXiv:1601.05588v1](https://arxiv.org/abs/1601.05588v1) and its checksum-pinned source bundle | Paper plus `supl-mkaf16.m`; the bundle contains the TeX, bibliography, and symbolic Mathematica supplement | No independently emitted numerical modes or polarizations; no finite-`f_ref` table; no native signed-mode projection; no cutoff-boundary samples; the [arXiv license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/) grants distribution to arXiv, not a downstream MIT grant |
| [Mezzasoma and Yunes, arXiv:2203.15934v1](https://arxiv.org/abs/2203.15934v1) and its source bundle | Paper formulas; the bundle contains TeX and bibliography only | No numerical data or executable; the GR amplitude review used here is nonspinning; no finite-`f_ref` and per-mode-cutoff oracle |
| [LALSuite TaylorF2 public documentation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_inspiral_taylor_x_x__c.html) and the maintained public-API probe | One TaylorF2 frequency series with a documented finite `f_ref`; public mode dispatch does not provide this eight-mode TaylorF2 model | Not an all-eight independent mode record; LAL source/copies and fitted adapters were outside the acceptance boundary |
| [WF4Py](https://github.com/CosmoStatGW/WF4Py) and [PyCBC waveform interface](https://github.com/gwastro/pycbc/blob/master/pycbc/waveform/waveform.py) | Restricted TaylorF2 in WF4Py; a LAL-backed interface in PyCBC | Neither supplies an independent all-eight oracle with the required options |

Registry/repository checks used both arXiv identifiers, titles and DOIs.
[Zenodo](https://zenodo.org/) returned no matching records; GitHub repository
search returned no exact matches; Crossref exposed no related data/code
objects; DataCite returned only the arXiv DOI records. These negative results
are dated observations, not proof that no future or unindexed oracle exists.

## Frozen fixture provenance

`tests/data/taylorf2_hm_mishra2016_samples.json` stores generated amplitude
ratios and a normalized nonspinning `TF28Phasing`. The latter is a symbolic
transformation of checksum-pinned `supl-mkaf16.m`, not an independently
authored numerical output. The original `.m` file is absent, but that does not
make the transformed expression MIT work. Because no explicit downstream
redistribution or MIT grant was found, the fixture now records its terms as
unresolved and `THIRD_PARTY_NOTICES.md` excludes it from the repository's MIT
claim.
