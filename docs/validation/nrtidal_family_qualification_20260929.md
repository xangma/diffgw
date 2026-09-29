# NRTidal family qualification boundary — 29 September 2026

**Disposition:** The scoped v1 D, v2 D and conditional v2 Pv2 mathematics,
Torch and JAX gates pass. This receipt does not qualify a public precessing
BNS waveform or resolve the known source versus pinned-binary conflicts.
The native implementation was unchanged in this audit. The added executable
evidence is an original-source diagnostic and derivative/JIT checks of the
existing composition paths.

## Versioned sources and public-code comparison

The comparison records *offered options*, not equivalence inferred from
approximant names. The paper contracts come from [Dietrich et al.,
arXiv:1706.02969v1](https://arxiv.org/html/1706.02969v1), its
[v2 revision](https://arxiv.org/html/1706.02969v2), the later
[merger-fit paper](https://arxiv.org/html/1804.02235), and
[Dietrich et al., arXiv:1905.06011v2](https://arxiv.org/html/1905.06011v2).
Public-code offerings were checked against [LALSimulation 7.26 NRTidal
documentation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_i_m_r_t_i_d_a_l__c.html),
[LALSimulation 7.26 Pv2 documentation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_phenom_p_8c.html),
the [PyCBC waveform list](https://pycbc.org/pycbc/latest/html/waveform.html),
and the [Ripple model list](https://github.com/tedwards2412/ripple)
on 29 September 2026. The numerical oracle used here was the installed
LALSuite **7.26.15** distribution, not LAL implementation source.

| Family and diffgw option | Published contract and native evidence | Other public-code offering and remaining gap |
| --- | --- | --- |
| `IMRPhenomD_NRTidal` v1 | The pinned [v2 revision](../math/nrtidalv1-phase.md) uses effective tidal coupling and its Eq. (7) Padé phase. The first arXiv revision prints a different $\kappa_2^T$ mass weighting in Eq. (2); the later standard $\kappa_2^T$ also differs from that original printed expression. The source diagnostic now demonstrates equal-mass agreement and unequal-mass divergence. Native Torch/JAX phase, taper and LAL binary gates pass to their existing bounded contracts. | LALSimulation offers `NRTidal_V` for D and PyCBC lists `IMRPhenomD_NRTidal`. Those names do not identify which printed coupling revision or every internal coefficient. diffgw does not expose the original v1 printed coupling as a waveform option; its historical taper conventions retain their documented bounds. |
| `IMRPhenomD_NRTidalv2`, `binary_compatible` | The default retains the historically observed D tidal amplitude and spin fit. It has pinned-binary premerger agreement, including the [composed holdout](nrtidalv2_composed_holdout_20260927.md), but the amplitude normalization is not derived from 2019 Eq. (24). The nonzero-tide taper binary comparisons remain expected failures. | LALSimulation offers D `NRTidalv2_V` and `NRTidalv2NoAmpCorr_V`; PyCBC lists `IMRPhenomD_NRTidalv2`. These offerings do not establish that the default diffgw option is paper-equivalent. Ripple's current README lists D NRTidalv2 with verification limited to low spins, $\chi_{1,2}<0.05$; that claim does not qualify this option or its high-spin/taper behavior. |
| `IMRPhenomD_NRTidalv2`, `paper_2019` | [2019 Table II](../math/nrtidalv2-paper-d-composition.md) assigns D no Eq. (24) tidal amplitude. The scalar reference composes the printed tidal and 2PN+3PN self-spin phases and taper. Torch/JAX complex quotients pass, as do the new 300 Hz phase-derivative and JAX JIT gates. This is a paper-math variant, with no claim of exact LAL binary parity. | LAL's `NRTidalv2NoAmpCorr_V` is a named no-amplitude option, but its full phase, spin and taper equivalence to `paper_2019` was not established. Ripple's listed D v2 model does not specify this diffgw option. |
| `IMRPhenomPv2_NRTidalv2`, conditional aligned carrier | [2019 Eqs. (17)–(29) and Sec. III.4](../math/nrtidalv2-pv2-aligned-carrier.md) add the signed Eq. (24) amplitude and 2PN+3PN+3.5PN self-spin before Pv2 rotation. An independent scalar reference checks native Torch/JAX finite-size assembly from a supplied BBH carrier, and the new fixed-carrier value/gradient/JAX JIT gate removes BBH-generator dependence from that check. Supplied-angle projection is separately gated. | LALSimulation offers `IMRPhenomPv2_NRTidalv2`; PyCBC lists it. diffgw still has no public Pv2 BNS dispatch. BNS angle evolution, phase/time origin and reference-frequency anchoring remain [pending contracts](../math/nrtidalv2-pv2-public-assembly-boundary.md); conditional assembly does not establish full waveform parity. |
| `IMRPhenomPv2_NRTidal` v1 | No diffgw v1 Pv2 paper-math or native composition gate was added. | LALSimulation and PyCBC list this v1 Pv2 approximant. It remains a diffgw family capability gap. |

PyCBC's list is evidence of exposed names, not a separately tested
implementation. Ripple's README lists D v2, with its stated low-spin
verification limit; it does not list v1 or Pv2 NRTidal in the checked model
list. No cross-code equality is claimed from those lists.

## Provenance of the executable evidence

The independent scalar modules under `src/diffgw_reference/` use the cited
paper equations with SymPy and high-precision mpmath. The native Torch/JAX
approximants do not call those references. The new v1 diagnostic in
`nrtidalv1_phase.py` is explicitly a *printed original-source audit*, not
a third native coupling branch. `test_nrtidalv1_phase_math_reference.py`
checks source-revision mass weighting and body exchange.
`test_nrtidalv2_paper_d_composition_math_reference.py` differentiates the
paper D quotient phase, and
`test_nrtidalv2_pv2_aligned_carrier_math_reference.py` compares both native
backends with an 80-digit scalar composition using a fixed complex128 BBH
input. Both added derivative gates cover JAX CPU FP64 JIT; their limits are
frozen in the linked math specifications. This audit introduced no LAL
constants, residual fit or new native waveform branch.

## Execution and observed result

Host `len`, owned scratch cwd
`/home/xangma/tmp/diffgw-nrtidal-family-qualification-20260929`, Linux CPU
FP64. The scratch copy came from isolated branch
`codex/nrtidal-family-qualification`. Python was
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python` (3.12.3), with
Torch 2.14.0+cpu, JAX 0.11.2, SymPy 1.14.0, mpmath 1.3.0 and LALSuite
7.26.15. The strict scoped command was:

```sh
DIFFGW_REQUIRE_LAL_BINARY=1 bash run_focus.sh
```

`run_focus.sh` set `PYTHONPATH=src:tests`, `JAX_ENABLE_X64=1` and
`OMP_NUM_THREADS=1`, then ran 18 files covering NRTidal v1/v2 phase,
amplitude, spin, taper, paper-D, Pv2 conditional, LAL-binary and parity
checks. Its retained log is `focus_final.log` in the scratch cwd.
**Result: 261 passed, 18 xfailed, 0 skipped in 81.57 s.** The 18 expected
failures predate this change and remain strict, especially the nonzero-tide
v2 D taper binary residuals of **0.000153757344** (equal mass) and
**0.000189164604** (unequal mass) against the frozen **0.00005** bound.
After the final small source-assertion and test-import edits, a strict rerun
of their two affected files returned **24 passed in 16.64 s**. All SSH
test processes exited; no running PID or stop action remains.

The existing [D provenance audit](nrtidalv2_provenance_20260927.md)
records the unresolved paper/binary amplitude normalization, Love–Q and
printed merger-fit conflicts. The [Pv2 public-assembly boundary](../math/nrtidalv2-pv2-public-assembly-boundary.md)
defines the source evidence needed before a complete precessing BNS claim.
The next bounded discriminator for the D taper is the read-only
multi-release, multi-tide binary comparison specified in that provenance
audit; no coefficient should be inferred from a residual alone.
