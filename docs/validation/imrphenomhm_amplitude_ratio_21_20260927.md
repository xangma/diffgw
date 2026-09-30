# IMRPhenomHM (2,1) 1.5PN amplitude-ratio gate — 27 September 2026

The [six-section source specification](../math/imrphenomhm-amplitude-ratio-21.md)
was frozen at `cbfe4bb`, before implementation or numerical comparison.
[London et al., arXiv:1708.00404v4, Eqs. (3)–(4) and the paragraph after
Eq. (4)](https://arxiv.org/pdf/1708.00404v4) prescribe a (2,1) SPA ratio
through relative 1.5PN. [Mishra et al., arXiv:1601.05588v1, Eqs. (7),
(11), (12a), (12b)](https://arxiv.org/pdf/1601.05588v1) fix the complete
complex polynomial through $V_1^4$. The independent SymPy source reduces
the full SPA quotient to $(\sqrt2/3)P(v_g)P(v_x)/P(v_{2x})$ exactly; mpmath
evaluates its magnitude and analytic one-sided Eq. (5) slopes. The test
divides the public unprojected (2,1) mode magnitude by the separately
evaluated public PhenomD carrier magnitude at the independently mapped
frequency. Code and gates are in `91c7c19`.

The source coefficient of $V_1^4$ is
$\chi_a(4771/1344-11941\eta/336)+\delta\chi_s(4771/1344-2549\eta/336)
+\delta[-\pi-i(1/2+2\log2)]$. With $\delta^2=1-4\eta$, both native
production implementations had used `c_tail + c_spa - c_td`, yielding
$\chi_a(3427/1344-2101\eta/336)+\delta\chi_s(3427/1344-965\eta/336)$
for the spin terms. The initial correction changed `- c_td` to `+ c_td`
in both frontends. Exact-head CI run
[36294348734](https://github.com/xangma/diffgw/actions/runs/36294348734)
then failed the strict binary-oracle gate only for
`IMRPhenomHM_diagnostic` and `IMRPhenomHM_holdout_5`; all other catalog
and holdout cases passed. The printed equation and the established binary
behavior therefore need distinct claims. The public default now keeps
`- c_td` as `hm_21_pn_variant="binary_compatible"`, while
`hm_21_pn_variant="mishra_eq12b"` selects the printed `+ c_td` coefficient.
The source gate explicitly selects the latter. Neither source coefficient,
map, D carrier, nor tolerance changed.

Host `len`; original base `b111a76`; isolated branch
`codex/hm-leading-44-amplitude`; source copied to
`/home/xangma/tmp/diffgw-hm21-source-gate-20260927`. Python:
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`; Python 3.12,
mpmath 1.3.0, SymPy 1.14.0, NumPy 2.5.3, Torch 2.14.0+cpu,
JAX 0.11.2, pytest 9.1.1, and PyCBC 2.11.0 from the compatible package
directory. Every command used this prefix from its stated cwd:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
PYTHONPATH=/home/xangma/tmp/diffgw-pycbc-2.11.0-full/packages:src:tests \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q
```

The **pre-fix red gate** ran with the original native source and the new
independent reference/tests, before the sign change. Exact suffix:
`tests/test_imrphenomhm_amplitude_ratio_math_reference.py
--reference-device=cpu -ra -k beta21 -x`. The original log is
`/home/xangma/tmp/diffgw-hm21-source-gate-20260927/hm21-red.log`.
For explicit process status the same pre-fix tree was replayed in
`/home/xangma/tmp/diffgw-hm21-red-replay-20260927` with that exact
suffix and log `hm21-red-replay.log`: **exit 1, 1 passed, 1 failed,
15 deselected in 24.35 s**. The source reduction/scalar test passed.
The first public failure was Torch, `phenomd_runtime`,
$(m_1,m_2,\chi_1,\chi_2)=(35,20,0.2,-0.1)$: all nine points failed,
maximum absolute discrepancy `0.05467972`, maximum relative discrepancy
`0.57887378`; the first measured/expected ratio was
`0.031046`/`0.025903`.

After the initial two-line correction, the exact ratio-file suffix was
`tests/test_imrphenomhm_amplitude_ratio_math_reference.py
--reference-device=cpu -ra`. Log:
`/home/xangma/tmp/diffgw-hm21-source-gate-20260927/hm21-green.log`;
**32 passed in 44.14 s**, no skips or warnings. This was before the
binary-compatible default was restored; the tests now select
`mishra_eq12b` explicitly. The independent source
reduction and printed coefficients are exact symbolic checks. For the
public quotient, the frozen tolerance is `rtol=1e-10, atol=1e-12`;
for one-sided join secants at geometric step `1e-7`, it is
`rtol=5e-5, atol=5e-8`. Both Torch and JAX passed three mass/spin
cases and both map settings. In particular, equal masses with unequal
spins have nonzero ratios. At equal mass and equal spin, literal Eq. (4)
is $0/0$ and the scalar rejects it; the public (2,1) mode remained
finite and identically zero across both map settings and frontends.
The zero-mode branch is checked, but no path-independent quotient limit
is claimed.

The neighboring HM suffix was
`tests/test_imrphenomhm_frequency_map_math_reference.py
tests/test_imrphenomhm_phase_offsets_math_reference.py
tests/test_imrphenom_hm.py tests/test_phenomhm_reconstruction.py
--reference-device=cpu -ra`. Log:
`/home/xangma/tmp/diffgw-hm21-source-gate-20260927/hm21-neighbors.log`;
**exit 0, 88 passed in 27.64 s**, no skips, one Torch JIT deprecation
warning.

This is source fidelity for the printed 1.5PN (2,1) amplitude ratio and
sampled implementation agreement through the public opt-in mode. The public
quotient necessarily uses the native D carrier and QNM frequencies as
supplied inputs; it is not an independent qualification of either.
Higher PN terms, odd-mode phase/parity, projection, LAL agreement,
physical accuracy, and the complete HM waveform remain unqualified.
No LAL code, constants, or binary output entered these tests.

## Binary-compatible default and explicit paper option

The integrated checkout at `9cbb2a2` initially put the printed coefficient
on the default path. The strict binary oracle in CI run 36294348734 failed
only `IMRPhenomHM_diagnostic` and `IMRPhenomHM_holdout_5` (18/19 catalog
and 15/16 holdout cases passed). The subsequent remedy restored the old
`- c_td` default and named the printed `+ c_td` path `mishra_eq12b`; the
source gate explicitly selects the latter. This preserves the existing
binary and paper gates with their original tolerances.

The remedy was checked on `len` in the isolated tree
`/home/xangma/tmp/diffgw-hm21-dual-variant-20260927`, using the environment
prefix above. The amplitude-ratio file passed **34 tests in 47.93 s**,
including a new public default-versus-explicit-default equality and paper
option distinction for Torch and JAX; log `hm21-dual-source.log`. Direct
strict replay through the frozen parity harness passed both formerly failing
cases; log `hm21-dual-binary.log`. The diagnostic plus/cross complex L2
errors were `1.0523665e-7`/`1.0669251e-7`; holdout_5 errors were
`5.5745474e-7`/`5.4138847e-7`, all below the unchanged `1e-6` gate.
The corresponding PSD mismatches were at most `1.6432e-14` and
`1.1969e-12`, below the unchanged `1e-8` gate. This replay did not rerun
the whole strict catalog; the earlier CI run already showed its other
cases passing at the same integrated head.
