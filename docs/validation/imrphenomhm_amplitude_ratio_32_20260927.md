# IMRPhenomHM (3,2) 1.5PN amplitude-ratio gate — 27 September 2026

The [six-section source specification](../math/imrphenomhm-amplitude-ratio-32.md)
was frozen at `7820d13` before reference, tests, or native changes.
[London et al., arXiv:1708.00404v4, Eqs. (3)–(4) and the paragraph after
Eq. (4)](https://arxiv.org/pdf/1708.00404v4) prescribe the spin-dependent
1.5PN factor for $|m|=\ell-1$.
[Mishra et al., arXiv:1601.05588v1, Eqs. (7), (11), (12a),
(12d)](https://arxiv.org/pdf/1601.05588v1) give its SPA envelope. Because
$m=2$, the second quotient in London's Eq. (4) is unity wherever defined;
the full SPA powers cancel in the first quotient. The resulting magnitude is
$\beta_{32}=(\sqrt{5/7}/3)| (1-3\eta)v_g^2+4\eta\chi_s v_g^3|$,
$v_g=(\pi g)^{1/3}$. SymPy reduces the full quotient, and mpmath evaluates
the magnitude and analytic one-sided Eq. (5) slopes independently of native
code and LAL.

The production-called Torch and JAX `_production_hhat` paths previously
used only $(1-3\eta)v^2$ for (3,2); the similarly named public helper was
not on this path. The new public `hm_32_pn_variant="mishra_eq12d"` selects
the printed spin term with the D carrier. The default
`hm_32_pn_variant="binary_compatible"` keeps the original operation and
binary behavior. Both frontends reject unknown variants and the paper
variant with a non-D carrier. The public gate divides the unprojected
(3,2) mode magnitude by a separately evaluated public PhenomD carrier at
the independently mapped frequency.

Host `len`; source base `3ac8cc3`; isolated branch `codex/hm32-amplitude`.
Python `/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`: 3.12.3,
NumPy 2.3.5, Torch 2.14.0+cpu, JAX 0.11.2, SymPy 1.14.0, mpmath 1.3.0,
pytest 9.1.1, and PyCBC 2.11.0. Unless stated otherwise, commands ran on
CPU with this environment prefix from the stated checkout:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
PYTHONPATH=/home/xangma/tmp/diffgw-pycbc-2.11.0-full/packages:src:tests \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q
```

The **pre-change red snapshot** was
`/home/xangma/tmp/diffgw-hm32-red-20260927` and contained the new
reference/tests with unchanged native files. Command suffix:
`tests/test_imrphenomhm_amplitude_ratio_math_reference.py
--reference-device=cpu -ra -k beta32 -x`; log `hm32-red.log`.
**Exit 1: 1 source test passed, then the first Torch public test failed**
because `IMRPhenomHM.forward()` did not yet accept `hm_32_pn_variant`.
That red result establishes the absent public production option, not a
numerical residual.

The **post-change green snapshot** was
`/home/xangma/tmp/diffgw-hm32-green-20260927`. The same suffix without
`-x` passed **15 tests, 34 deselected in 17.19 s** after adding a scalar
negative-brace absolute-value check; log `hm32-green-final.log`. There
were no skips or warnings. This includes exact symbolic/source checks,
three mass/spin rows, both Eq. (5) transition choices, Torch and JAX
public values across both joins, analytic one-sided slopes, and exact
default-versus-explicit-default equality. The frozen public tolerances
were `rtol=1e-10, atol=1e-12` for values and `rtol=5e-5, atol=5e-8`
for join secants at geometric step $10^{-7}$.

The neighboring suite suffix was
`tests/test_imrphenomhm_amplitude_ratio_math_reference.py
tests/test_imrphenomhm_frequency_map_math_reference.py
tests/test_imrphenomhm_phase_offsets_math_reference.py
tests/test_imrphenom_hm.py tests/test_phenomhm_reconstruction.py
--reference-device=cpu -ra`; log `hm32-neighbors.log` in the green
snapshot. It passed **137 tests in 72.68 s**, with one existing Torch JIT
deprecation warning and no skips. That run preceded the final scalar-only
negative-brace assertion, which the targeted green replay covered.

The frozen strict binary harness was replayed from the green snapshot
using `/tmp/hm32_binary_gate.py`; log `hm32-binary.log`. Both targeted
default-mode cases passed: `IMRPhenomHM_diagnostic` plus/cross complex L2
errors $1.0523665\times10^{-7}$/$1.0669251\times10^{-7}$ and
`IMRPhenomHM_holdout_5` errors
$5.5745474\times10^{-7}$/$5.4138847\times10^{-7}$, below the unchanged
$10^{-6}$ gate. This is a targeted replay, not a whole-catalog run.

The gate establishes source fidelity for the printed (3,2) amplitude
factor and sampled public implementation alignment under the explicit
paper option. It does not independently qualify the supplied D carrier,
QNM frequencies, map construction, higher PN orders, phase/parity,
projection, physical accuracy, LAL agreement, or the complete HM waveform.
No LAL code, constants, or binary output entered the equation reference.
