# NRTidalv2 self-spin paper-D component receipt — 2026-09-27

**Scope:** Dietrich et al. (2019) printed Eqs. (26)–(29) and Table II for
positive $\Lambda$, aligned spins, CPU FP64, 2PN+3PN D self-spin. The separate
3.5PN component is tested as an equation, not included in the published D
composition. No pinned-binary or physical-data claim follows from this gate.

**Source:** [arXiv:1905.06011v2, Eqs. (26)–(29), Table II](https://arxiv.org/html/1905.06011).
The coefficient strings are transcribed at printed precision into an
independent mpmath reference. Torch/JAX do not import it. See the
[math specification](../math/nrtidalv2-self-spin.md) for conventions and cases.

**Frozen gate:** 30, 100, 300, 800 Hz, four mass/spin/deformability cases,
including body exchange and zero spin. Maximum component absolute error
$10^{-12}$; pre-taper complex quotient absolute error $10^{-10}$ without
alignment; spin derivative absolute error $10^{-10}$. Both coalescence and 30 Hz
reference-frequency adapters are checked. The latter is a native convention.

**Run:** host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-self-spin-20260927`, command
`PYTHONPATH=src:tests /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python
-m pytest -q tests/test_nrtidalv2_self_spin_math_reference.py
tests/test_nrtidal_reference.py tests/test_nrtidalv2_phase_math_reference.py
tests/test_nrtidalv2_amplitude_math_reference_native.py tests/test_nrtidal.py
tests/parity/test_nrtidal_parity.py`. Result: **116 passed, 1 skipped in
40.89 s**. Versions:
Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, pytest 9.1.1, mpmath 1.3.0.
The skip is an existing comparison requiring LALSimulation or PyCBC.

**Interpretation:** The printed equations and chosen D composition agree with
both native frontends on the frozen slice. This does not verify the whole
NRTidalv2 waveform or binary/NR parity. The paper's Fig. 7 text reports
$(C_Q,C_{Oc})=(4.30,7.28)$ at $\Lambda=350$, whereas its printed Eqs.
(28)–(29) yield $(5.29659,10.49832)$. The discrepancy remains unresolved;
this gate deliberately follows the numbered equations and records the
conflict. Historical binary-compatible CQ coefficients, Lambda clamps,
reference-frequency anchoring, merger taper, and full spinning-waveform
comparisons remain separate scopes.
