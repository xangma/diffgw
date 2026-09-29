# TaylorF2_HM minimal multimode oracle boundary — 2026-09-27

## Candidate and decision

The smallest useful *external waveform* gate is the nonspinning, unwindowed
`(2,2)+(3,3)` composition at unequal mass. The `(3,3)` leading amplitude is
proportional to the signed mass difference in [Mishra et al., Eqs. (9)–(12h)](https://arxiv.org/html/1601.05588)
and [Mezzasoma and Yunes, Eqs. (20)–(21i)](https://arxiv.org/html/2203.15934),
so unequal masses distinguish this candidate from a dominant-only carrier.
Use `chi1=chi2=0`, `f_ref=None`, and `f_isco_cutoff=False`, and require direct
complex per-mode `(h_plus,h_cross)` and their sum with pinned Fourier,
polarization, phase, time, distance, and angular conventions. The previously
declared maximum normalized complex error is `4e-10` in FP64 for each mode and
sum, with no fitted alignment. This two-mode gate would establish only that
slice, not the other six modes or any spin/reference-frequency/cutoff option.

No external same-model `(2,2)+(3,3)` complex waveform output was found in the
reviewed sources. The checksum-pinned [Mishra author supplement](taylorf2-hm-oracle-search-20260926.md)
provides symbolic expressions, not a separately emitted numerical series;
the maintained source-composed test already exercises **all eight** leading
nonspinning modes and summed polarizations in both frontends. Re-evaluating
that composition for just two modes cannot supply a new independent oracle.
The public [LAL TaylorF2 documentation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_inspiral__h.html)
describes a frequency-domain TaylorF2 waveform, but the pinned binary returns
one carrier series and offers no TaylorF2 FD-mode generator. IMRPhenomXHM and
TEOBResumSPA are different waveform models, so their higher-mode values cannot
serve as binwise TaylorF2_HM parity targets. This is a dated boundary, not a
claim that no future independent release can exist.

## Reproducible public-binary check

On `len` (Linux x86_64), the private staging copy was
`/home/xangma/tmp/diffgw-tf2hm-minimal-oracle-20260927`. The environment was
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python` with LAL 7.7.1,
LALSimulation 6.2.1, Torch 2.14.0+cpu, JAX/JAXLIB 0.11.2, SymPy 1.14.0,
and mpmath 1.3.0. `tests/test_taylorf2_hm_external_api.py` now uses unequal
`(1.7,1.2)` solar-mass components. At `30:0.25:500` Hz, the public
`SimInspiralTaylorF2` output has 2001 bins; inserting mode arrays `[(2,2)]`,
`[(2,2),(3,3)]`, or `[(2,2),(3,2)]` leaves that series exactly unchanged
(`max |delta| = 0` for each). The separate public
`SimInspiralChooseFDModes` request for TaylorF2 raises a generator/argument
error. These observations verify the API boundary, not a two-mode waveform.

The command below passed **54/54** on `len`, with no skips or expected failures:

```sh
cd /home/xangma/tmp/diffgw-tf2hm-minimal-oracle-20260927
env PYTHONPATH=src:tests:. JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -ra \
  tests/test_taylorf2_hm_external_api.py \
  tests/test_taylorf2_hm_author_supplement_math_reference.py \
  tests/test_taylorf2_hm_math_reference.py \
  tests/test_taylorf2_hm_paper_math_reference.py tests/test_oracle_ci.py
```

The external multimode verdict remains `UNRESOLVED_ORACLE`. A qualifying
future release would need a reproducible, independently maintained same-model
complex series or published numerical table with sufficient precision and
explicit redistribution terms. Its two mode values and sum must be compared
directly against both frontends at fixed physical inputs and conventions.
