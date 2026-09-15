# IMRPhenomHM transition-frequency convention audit

Date: 2026-09-13. This is a bounded clean-room review of the two transition
frequencies used by the current TorchWave `IMRPhenomHM` path. No LALSuite or
PyCBC implementation source was inspected, and no waveform-oracle residual was
used to select a convention.

## Primary-source evidence

The higher-mode mapping paper is explicit. London et al.,
[arXiv:1708.00404](https://arxiv.org/html/1708.00404), states after its
frequency-mapping equations that

```text
f_0^A   = 0.018 f_lm^RD / f_22^RD
f_0^phi = 0.014 f_lm^RD / f_22^RD.
```

The same assignment is present in versions 1, 2, 3, and 4 of the arXiv
manuscript. The paper's superscripts refer to the higher-mode amplitude and
phase mappings, respectively; its notation does not provide a basis for
swapping those labels.

The carrier model has a different pair of joins. Khan et al.,
[arXiv:1508.07253](https://arxiv.org/html/1508.07253), places the PhenomD
amplitude intermediate collocation/join at `Mf=0.014` and the inspiral-to-
intermediate phase join at `Mf=0.018`. Those values are also described as
separate amplitude and phase transitions in the PhenomD construction. This is
the convention currently used by TorchWave's carrier-frequency path:

```text
runtime amplitude join = 0.014 f_lm^RD / f_22^RD
runtime phase join     = 0.018 f_lm^RD / f_22^RD.
```

The two papers therefore define two distinct pairs of constants. The London
values are optimized higher-mode mapping transitions; the PhenomD values are
the carrier model's own amplitude and phase joins. A search of the primary
paper's version history and the available publication metadata found no
erratum or notation note that identifies the London values as reversed.

## Runtime decision

There is no unique source-defined correction that changes the current runtime
from the PhenomD carrier convention to London's separately stated higher-mode
convention, or vice versa. The better public-binary residuals of one choice are
not admissible evidence for changing a source-defined constant. This audit
therefore makes **no runtime change**.

The earlier `higher-modes.md` record described the London pair as the runtime
transitions. That description predates the current carrier-join implementation
and should not be read as evidence that the two source conventions are
interchangeable.

## Test audit

The former `test_hm_published_transitions_and_spa_rescaling` monkeypatched the
runtime mapping functions and then asserted London's `.018` amplitude and
`.014` phase values. It consequently failed against the current, intentionally
PhenomD-based runtime; the failure was a source-convention mismatch rather than
a mapping continuity failure. It is now named
`test_hm_runtime_transitions_and_spa_rescaling` and asserts the runtime pair,
while this report preserves the independently verified London values.

The phase-continuity test and phase-derivative test now sample the runtime
phase boundary at `Mf=0.018`. The generic mapping-continuity test continues to
exercise London's published mapping constants directly; it does not claim that
those constants are the current carrier joins.

Focused validation from the repository root:

```sh
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_phenomhm_reconstruction.py tests/test_clean_higher_modes.py
```

After the test-audit correction, all focused tests pass. `git diff --check`
also passes.
