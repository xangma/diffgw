# IMRPhenomHM (4,3) 1.5PN amplitude-ratio gate — 27 September 2026

The [six-section source specification](../math/imrphenomhm-amplitude-ratio-43.md)
was frozen at `dbf62cc`, before native implementation or numerical
comparison. [London et al., arXiv:1708.00404v4, Eqs. (3)–(5)](https://arxiv.org/pdf/1708.00404v4)
prescribe the restricted SPA amplitude ratio and mapped (2,2) carrier.
[Mishra et al., arXiv:1601.05588v1, Eqs. (7), (11), (12a), (12g)](https://arxiv.org/pdf/1601.05588v1)
give the (4,3) brace through $V_3^4$:
$P_{43}(v)=\delta(1-2\eta)v^3+(5/2)\eta(\chi_a-\delta\chi_s)v^4$.
The independently built SymPy Eq. (4) quotient reduces exactly to
$(3/4)\sqrt{3/35}\,P_{43}(V_3(g))P_{43}(V_3(x))/P_{43}(V_3(2x/3))$.
mpmath evaluates the magnitude and analytic one-sided map slopes. The
public gate divides each unprojected (4,3) mode magnitude by a separately
evaluated public PhenomD (2,2) carrier magnitude at the independent map.

Both native production paths previously omitted the printed spin term,
though the displayed `h_hat_multipole` helper included it. The explicit
`hm_43_pn_variant="mishra_eq12g"` now selects the printed brace. The
`"binary_compatible"` default retains the prior mass-only expression;
the test confirms an omitted option and explicit default give exactly
equal arrays, while the paper option changes a spinning case. A custom
(2,2) carrier cannot request the paper variant. No source equation,
mapping convention, or frozen tolerance was changed after the red gate.

Host `len`; local isolated worktree based on `3ac8cc3`; remote scratch
cwd `/home/xangma/tmp/diffgw-hm43-source-gate-20260927`. Python was
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python` (3.12.3), with
mpmath 1.3.0, SymPy 1.14.0, NumPy 2.5.3, Torch 2.14.0+cpu, JAX 0.11.2,
pytest 9.1.1, and the PyCBC 2.11.0 package directory. Test commands used:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
PYTHONPATH=/home/xangma/tmp/diffgw-pycbc-2.11.0-full/packages:src:tests \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
tests/test_imrphenomhm_amplitude_ratio_43_math_reference.py \
--reference-device=cpu -ra
```

The first pre-change run, with new tests but old native source, exited 1:
**19 failed, 1 passed** in 12.14 s. Its log is `hm43-red.log` in the
scratch cwd. Eighteen failures were the expected absence of the named
public option. One symbolic assertion used structural rather than algebraic
equality; this test assertion was corrected to expand the difference.
The scalar example and domain gate passed. To expose the underlying
physics mismatch separately, the same pre-change native files were run
with only the new option keyword removed from a scratch copy of the public
value test: `tests/test_imrphenomhm_amplitude_ratio_43_baseline.py
--reference-device=cpu -k values_and_one_sided`. Log
`hm43-red-physical.log`: **8 failed, 4 passed, 8 deselected** in 13.71 s.
Spinning unequal-mass and equal-mass unequal-spin cases failed in both
Torch and JAX and in both map settings. All four nonspinning cases passed.
At equal mass with unequal spin the old public ratio was zero at all nine
points, versus source values $0.000256$–$0.004812$ in the shown case.

With the explicit paper variant, the source file passed **20 tests in
19.98 s**, no skips; log `hm43-green.log`. The cases cover
$(m_1,m_2,\chi_1,\chi_2)=(35,20,0.2,-0.1)$, $(35,20,0,0)$, and
$(30,30,0.2,-0.1)$, both `phenomd_runtime` and `london_2018` maps,
Torch and JAX, below/at/between/above the two map joins. The independent
ratio tolerance was `rtol=1e-10, atol=1e-12`; one-sided secants used
geometric step $10^{-7}$ with `rtol=5e-5, atol=5e-8`. At equal mass
and equal spin, literal Eq. (4) is $0/0$ and the scalar rejects it; the
public mode remains finite and exactly zero in both backends and maps.

The neighboring suffix comprised
`tests/test_imrphenomhm_frequency_map_math_reference.py`,
`tests/test_imrphenomhm_phase_offsets_math_reference.py`,
`tests/test_imrphenomhm_amplitude_ratio_math_reference.py`,
`tests/test_imrphenom_hm.py`, and
`tests/test_phenomhm_reconstruction.py`, with
`--reference-device=cpu -ra`. Log `hm43-neighbors.log`:
**122 passed** in 67.22 s, one Torch JIT deprecation warning and no
skips. The final source-file rerun after a return-type cleanup passed
**20 tests** in 19.98 s, no skips; log `hm43-final.log`.

This is source fidelity and sampled implementation agreement for the
(4,3) amplitude factor with a public D carrier. It does not independently
qualify that carrier or QNM inputs. The phase, projection, whole waveform,
physical accuracy, and LAL parity remain separate claims. No LAL source,
constants, or binary output entered this gate.
