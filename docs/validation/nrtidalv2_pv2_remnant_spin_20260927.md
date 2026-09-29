# NRTidalv2 Pv2 inherited remnant-spin receipt — 27 September 2026

**Result:** The source-complete *in-plane remnant-spin map* passed its
independent scalar gate in Torch and JAX. Public IMRPhenomPv2_NRTidalv2
dispatch remains unqualified.

The [frozen specification](../math/nrtidalv2-pv2-remnant-spin.md) applies
[Dietrich et al. (2019), Sec. III.4](https://arxiv.org/html/1905.06011) to
the [Pv2 technical note, Eqs. (29)–(30), (46), (51)](https://dcc.ligo.org/public/0122/T1500602/004/PhenomPv2_technicalnotes.pdf).
The scalar reference independently evaluates the printed effective
in-plane-spin and remnant-spin equations with mpmath. The Torch and JAX
waveform drivers now call explicit helpers for the same remnant map; the
historical signed retrograde behavior is unchanged. The tested paper domain
has positive aligned final spin, so it makes no claim about that signed
extension.

Three BNS-scale configurations cover equal and unequal masses, both bodies
with nonzero in-plane spin, and either body dominating \(\chi_p\). For each,
the native \(\chi_p\) and remnant-spin value agree with the scalar result to
the frozen `2e-15` absolute bound in CPU FP64. Neither the aligned final-spin
fit nor the QNM frequencies are treated as independently proved by this gate.

Validation ran on `len`, scratch cwd
`/home/xangma/tmp/diffgw-nrtidalv2-remnant-20260927`, using the current
available environment `/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python`
(Torch 2.14.0+cpu, JAX 0.11.2, pytest 9.1.1, mpmath 1.3.0). Command:

```sh
PYTHONPATH=src /home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python -m pytest -q tests/test_nrtidalv2_pv2_remnant_spin_math_reference.py tests/test_phenompv2_waveform_clean.py tests/test_nrtidalv2_pv2_aligned_carrier_math_reference.py
```

`30 passed, 1 skipped in 11.66s`; log `pytest.log` in the scratch cwd.
After preserving the JAX driver's original multiplication order in the
extracted helper, the same command passed again (`30 passed, 1 skipped in
10.49s`).
The skipped existing Pv2 parity-harness test requires unavailable PyCBC;
`pytest -q -rs tests/test_phenompv2_waveform_clean.py` confirmed its reason
(`18 passed, 1 skipped`). No LAL source, fitted residual, phase alignment,
or waveform retuning entered this gate.

The next public-dispatch boundary is specific. The inherited Pv2 technical
note sets the longitudinal spin input for the NNLO angles to the larger
body's aligned spin in Eq. (32), while the native Torch/JAX drivers use a
mass-weighted mapping previously selected for BBH binary compatibility.
The note's Eq. (26) prints `3.373 - 19η/8 - η²/24` for the 2PN orbital
coefficient, while [Pratten et al. (2020), Eq. (27)](https://arxiv.org/html/2004.06503)
prints `27/8 - 19η/8 + η²/24`; the native code has the mixed
`27/8 - 19η/8 - η²/24`. Therefore an exact *paper* angle gate against the
current native drivers cannot pass without choosing and naming a convention.
Dietrich Sec. III.4 directs insertion before rotation, but does not say
whether the Pv2 remnant-frequency spline time shift is sampled from the BBH
or finite-size carrier, nor does its phase decomposition set the BNS
reference-frequency phase origin. Those choices need a separately frozen
contract before exposing a complete public BNS dispatch.
