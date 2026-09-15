# MIT calibration audit: NRTidal v1/v2

Date: 2026-09-13. Scope was limited to public primary scientific papers and
black-box inspection of the current TorchWave runtime. No LAL/PyCBC/TEOB
implementation source, NR data, oracle fitting, or the bodies of the legacy
fit modules were inspected.

## Public scientific values

The NRTidal v1 frequency-domain Padé values are printed in Dietrich, Bernuzzi
and Tichy, *Phys. Rev. D* 96, 121501 (2017), arXiv:1706.02969, Eq. 7:

```
n1=-17.428, n3/2=31.867, n2=-26.414, n5/2=62.362,
d1=-19.924, d3/2=36.089, c_newt=2.4375.
```

Dietrich et al., *Phys. Rev. D* 100, 044003 (2019), arXiv:1905.06011,
Eqs. 18–23, prints the NRTidalv2 frequency-domain fit values

```
n5/2=90.550822, n3=-60.253578, d1=-15.111208, d2=8.0641096,
c1=3115/1248, c3/2=-pi, c2=28024205/3302208, c5/2=-4283*pi/1092.
```

The other NRTidalv2 phase coefficients are constrained by the paper's Eq. 21
and are therefore reproducibly computed from those printed values. The same
paper's Eq. 24 gives the amplitude numerator coefficients
`449/108` and `22672/9`, and publishes the fitted denominator only as
`d=13477.8`.

Sources: [arXiv:1706.02969](https://arxiv.org/abs/1706.02969) and
[arXiv:1905.06011](https://arxiv.org/abs/1905.06011). These sources establish
the equations and displayed decimal values; they do not publish extra digits
for the fit parameters.

## Runtime comparison and decision

Black-box import of the current required symbol dictionaries showed that the
v1 values and all v2 phase values agree with the paper transcription (the
constrained v2 values evaluate to the current float64 values). The current
amplitude dictionary instead contains `d=13477.8073677`, while the paper only
supports `13477.8`. The additional seven decimal places have no public
paper-based provenance found in this audit.

The runtime now imports a new clean-room module containing these public
equations and values; the former fit module was moved outside the repository
to `/Users/xangma/.cache/torchwave/external-validation/nrtidal/`. Four
source-literal helper tests that hard-coded `13477.8073677` previously failed
against the paper's `13477.8`; these are scientific expectation tests rather
than frozen whole-waveform parity gates and were updated to the printed paper
value. The affected reconstruction and NSBH tests now pass (`15 passed`).

Strict black-box parity remains passing for all four affected cases: catalog
`IMRPhenomD_NRTidal` and `IMRPhenomD_NRTidalv2`, plus their holdouts. The
NRTidalv2 post-change maximums were complex relative L2 `2.72e-7` (catalog)
and `1.61e-7` (holdout), with magnitude relative L2 `7.19e-10` and
`1.22e-9`, respectively; all are below the unchanged `1e-6` gates. The
NRTidalv1 metrics are unchanged. These results use the native strict parity
harness and are not a license claim for unrelated PhenomD tables.

The paper-based reconstruction is scientifically sufficient to document the
model equations, but it is not sufficient to certify exact runtime parity or
the whole package's MIT status. The PhenomD source-table gap remains outside
this narrow audit.

Structured native results, including canonical input SHA-256 values, are in
[`verification/nrtidal/results.json`](verification/nrtidal/results.json). The
exact output path is
`/Users/xangma/repos/torchwave/docs/reviews/2026-09-13-parity-review/verification/nrtidal/results.json`.
They were generated with:

```
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  docs/reviews/2026-09-13-parity-review/verification/nrtidal/generate_comparison.py
```

The clean runtime source is included in both distributions; the legacy file
and all `.m` assets are excluded. The wheel check was run with
`/Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pip wheel . --no-deps
--wheel-dir /tmp/torchwave-mit-packaging-20260913.VpUuAo` and produced
`/tmp/torchwave-mit-packaging-20260913.VpUuAo/torchwave-0.1.0-py3-none-any.whl`
(SHA-256
`295858890ae5a5277a31d7f74b1da6f1472910421091e33f90da7d22f571fc4f`). The
archive contained neither `nrtidal_unrounded_fits.py` nor any `.m` file. The
setuptools sdist check produced
`/tmp/torchwave-mit-sdist-build-20260913.yRJjqB/torchwave-0.1.0.tar.gz`
(SHA-256
`b221ee5382884ac7c0e0fb79f7dcb538e3a4ed84acf9edde2920fbfba389c3df`), also
with neither forbidden file type. `pyproject.toml` declares MIT license
metadata and includes `LICENSE` plus `THIRD_PARTY_NOTICES.md`. This is a
packaging observation within this audit, not a legal certification of the
full source tree.
