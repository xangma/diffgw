# TaylorF2_HM independent-oracle search — 2026-09-26

## Outcome

No qualifying published numerical oracle was located. No external-parity test
or numerical claim was added. The existing author-source, paper-fidelity,
scalar-reference, Torch/JAX-parity, angular, and dominant-mode gates retain
their narrower labels.

This was a bounded clean-room search. A same-equation SymPy evaluation, a
translation of the Mishra supplement, a LAL-derived series, or a fit to diffGW
output was rejected as independent evidence.

## Reduced nonspinning gate

Restricting the claim to `chi1=chi2=0`, `f_ref=None`, and
`f_isco_cutoff=False` removes the spin-recipe, reference-frequency, and
per-mode-mask requirements below. It does **not** supply an independent
eight-mode numerical oracle. Before the diagnostic run, the candidate gate was
fixed to all eight listed modes, direct unaligned complex per-mode
$(h_+,h_\times)$ and their sum in FP64 Torch and JAX, and a maximum complex
error of $4\times10^{-10}$ per unmasked mode and sum, normalized by the
corresponding reference magnitude (or summed mode magnitudes for a polarization
sum). No fitted phase, time, amplitude, or frequency shift is permitted.
Concrete sample inputs are the three binaries and frequencies in
`tests/data/taylorf2_hm_mishra2016_samples.json`, with distance 410 Mpc,
$t_c=0.0041$ s, $\phi_c=0.23$, and orientations
$(\iota,\phi_0)=(0.91,0.37),(1.32,-0.44)$.

The existing author-supplement test already evaluates precisely this
nonspinning, unwindowed slice as a **source-composed** complex check: it
extracts the leading terms and nonspinning phase from the pinned symbolic
supplement, adapts their convention, and supplies the polarization projection
from García-Quirós et al. Eqs. (8)–(9). Its $8\times10^{-10}$ component
threshold establishes agreement with that composition, not independent
numerical parity. Re-evaluating those same expressions in another interpreter
would check parsing or arithmetic but would still share the symbolic source and
the locally supplied composition.

The [Mishra et al. author source](https://arxiv.org/abs/1601.05588v1)
contains symbolic amplitudes and phasing, but no separately emitted complex
mode or polarization table. The public [WF4Py restricted TaylorF2](https://wf4py.readthedocs.io/en/latest/waveforms.html)
and [gwfast feature inventory](https://github.com/CosmoStatGW/gwfast/blob/master/NEW_FEATURES.md)
give a dominant-only TaylorF2; gwfast's listed higher modes belong to its
different TEOBResumSPA model. LAL's public TaylorF2 API returns one carrier;
its FD-mode dispatch has no TaylorF2 generator, as checked by the maintained
`tests/test_taylorf2_hm_external_api.py` black-box test. Searches for
`TaylorF2HM`, `TaylorF2_HM`, the Mishra supplement identifier, and nonspinning
eight-mode TaylorF2 numerical data found no independently maintained exact
model. These are dated negative observations, not a proof of nonexistence.

Thus the reduced numerical-oracle gate remains open. A qualifying output may
omit spins, finite $f_{\rm ref}$, and cutoff samples, but must provide all eight
complex modes or both polarizations per mode and their sum with fixed physical
inputs, units, phase, Fourier, tetrad, and mode conventions. The comparison
must be direct and reproducible from a pinned, suitably redistributable
published table or independently maintained executable. The `(2,2)` LAL
carrier and angular-basis checks cannot substitute for the other seven modes.

On `len` (Linux x86_64), the existing bounded checks passed **46/46** in
24.79 s using Python 3.12.3, Torch 2.14.0+cu130 on CPU, JAX/JAXLIB 0.11.2,
SymPy 1.14.0, mpmath 1.3.0, LAL 7.7.1, and LALSimulation 6.2.1. The command
was `env PYTHONPATH=src:. JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest -q -ra` followed by
`tests/test_taylorf2_hm_author_supplement_math_reference.py`,
`tests/test_taylorf2_hm_math_reference.py`,
`tests/test_taylorf2_hm_paper_math_reference.py`,
`tests/test_taylorf2_hm_external_api.py`, and `tests/test_oracle_ci.py` in the
private staging checkout `/home/xangma/tmp/diffgw-tf2hm-nonspin-oracle-20260926`.
No new numerical gate, fixture, or verification claim was added.

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
