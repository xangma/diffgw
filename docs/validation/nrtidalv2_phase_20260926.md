# NRTidalv2 nonspinning tidal-phase qualification — 26 September 2026

**Outcome:** PASS, 26/26 tests, no skips, expected failures or deselection.
This receipt concerns only [the phase slice](../math/nrtidalv2-phase.md),
not the complete NRTidalv2 waveform or physical model accuracy. The source
branch was based on `13b82672f3342c92e07bd93abd54eecb19c11fb4`.

The test ran on `len`, Linux 6.8.0-138-generic x86_64, in
`/home/xangma/tmp/diffgw-nrtidalv2-phase-20260926-e4a2`. The worktree
was copied there without altering `len`'s existing repositories. The command
was:

```sh
DIFFGW_REQUIRE_LAL_BINARY=1 OMP_NUM_THREADS=1 \
PYTHONPATH=/home/xangma/tmp/diffgw-nrtidalv2-phase-20260926-e4a2/src:/home/xangma/diffgw-pr45-318b37f7/.oracle-deps \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -s \
  tests/test_nrtidalv2_phase_math_reference.py \
  tests/test_nrtidalv2_phase_lal_binary.py
```

The complete output is in `qualification.log` in that scratch directory.
The first background run had Python worker PID `3789013` and passed in
18.06 seconds. After making missing dependencies fatal in strict mode, the
same focused gate passed again in 17.24 seconds; the log contains this final
run.
Python was 3.12.3; Torch 2.14.0+cpu, JAX 0.11.2, SymPy 1.14.0,
mpmath 1.3.0, and NumPy 2.5.3. The installed LALSuite distribution was
**7.26.15** (`lal` 7.7.1, `lalsimulation` 6.2.1). The public
`lalsimulation` extension used by the run was
`_lalsimulation.cpython-312-x86_64-linux-gnu.so` with SHA-256
`e816879d23e652e8b3928496d6fd261de0832b5c3781c50e33a458042495b42e`.
No LAL source was inspected.

The paper-reference gate checked the printed Padé fit inputs and PN expansion,
body exchange, zero tides, both frontend unit-complex quotients at 30, 100,
300, 500 and 800 Hz, and autodifferentiated frequency derivatives at 300 Hz.
All met the predeclared phase and derivative tolerances in the specification.

The binary gate compared *unaligned*, normalized complex
`IMRPhenomD_NRTidalv2 / IMRPhenomD` quotients at 30–800 Hz in 5 Hz steps.
The limits were relative complex $L_2\le10^{-6}$ and maximum pointwise
phase residual $\le10^{-5}$ rad. Observed LAL-to-paper results were:

| Case | Relative complex $L_2$ | Maximum phase residual (rad) |
| --- | ---: | ---: |
| Equal mass | `1.03826464e-8` | `3.29739276e-8` |
| Unequal mass | `1.24022574e-8` | `3.93515773e-8` |
| Exchanged bodies | `1.24022574e-8` | `3.93515773e-8` |
| Zero tides | `1.09702555e-17` | `6.33135952e-17` |

The largest Torch/JAX-to-LAL relative complex $L_2$ was
`1.24022573e-8`, and the largest maximum phase residual was
`3.93515742e-8` rad. The production merger-frequency fit gives at least
1709 Hz for these four systems, so 800 Hz lies below its taper onset.

These checks establish the printed phase expression and agreement with the
named binary on the declared cases. They do not validate the NR fit against
physical data or qualify the amplitude, self-spin, merger/taper, other modes,
nonzero spins, other domains, or inherited point-particle model.
