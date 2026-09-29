# Newtonian eccentric EMRI source strain evidence — 2026-09-29

## Executed environment and results

Host `len` (Linux), scratch cwd `/home/xangma/diffgw-emri-scope-20260929`, base checkout `64e6661` plus the scoped files in this commit, venv Python `/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`. Versions: PyTorch `2.14.0+cpu`, JAX `0.11.2`, SymPy `1.14.0`, mpmath `1.3.0`, NumPy `2.5.3`, pytest `9.1.1`. JAX x64 is enabled by the test module. Only CPU float64 results are reported.

| Check | Reproduction command after `cd` to the scratch cwd | Observed result | Remote log |
| --- | --- | --- | --- |
| Paper-derived reference, both native frontends, and existing Peters flux | `PYTHONPATH=src /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q tests/test_emri_newtonian_quadrupole_reference.py tests/test_emri_peters_flux_math_reference.py` | `20 passed in 22.90s` | `emri-focused-final.log` |
| Legacy EMRI API and specialized EMRI parity | `PYTHONPATH=src /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q tests/test_emri_kludge.py tests/parity/test_specialized_parity.py -k EMRI` | `8 passed, 277 deselected in 10.67s` | `emri-legacy-final.log` |

The focused reference covers exact SymPy circular identities; mpmath source strain at `e=0,0.2,0.55`, fixed observer angles and sampled times; independent high-precision five-point derivatives in `p0,e0,iota,gamma`; Torch batch values; JAX batched `jit`; RK4 grid stability; and the Bessel truncation edge `N=8,e=0.6,p0=12.1`. Relative/absolute value tolerances are `3e-12/2e-34`; derivative tolerances are `4e-10/1e-34`. These are sampled implementation comparisons, not a uniform error certificate. The initial development run failed five checks because `mp.diff` drove perturbations beneath the fixed mpmath evaluator precision and a convergence-ratio assertion sampled a floating-point plateau. The independent gradient oracle was changed to an explicit high-precision five-point stencil, and the grid check now asserts stability at `1e-12`; the final gates above passed.

## Source and implementation provenance

`src/diffgw_reference/emri_newtonian_quadrupole.py` defines `emri-newtonian-eccentric-quadrupole/v1` from [Peters (1964), Eqs. (5.6)–(5.7)](https://doi.org/10.1103/PhysRev.136.B1224) and [Barack–Cutler (2004), Sec. III B, Eqs. (2)–(10), (27)](https://arxiv.org/pdf/gr-qc/0310125). Its symbolic flux dependency is the independent `emri-peters-leading-flux/v1` reference. The reference uses SymPy expressions, mpmath Bessel evaluation, and high-precision fixed-grid RK4. Native Torch and JAX use their own tensor code and a 21-term integer Bessel series. Both native strains match this *same finite Newtonian option* at the sampled values. The common RK4 stepping choice leaves shared integration truncation error possible.

## Public model comparison and open gates

[FastEMRIWaveforms v2.1.1](https://fastemriwaveforms.readthedocs.io/en/v2.1.1/user/main.html) exposes separate Schwarzschild eccentric flux, Kerr eccentric equatorial flux, and PN5 AAK waveform options; its [trajectory guide](https://fastemriwaveforms.readthedocs.io/en/v2.1.1/tutorial/Trajectory_tutorial.html) describes their distinct inspiral equations. The diffgw definition checked here has Newtonian Peters flux, Kepler phase, finite quadrupole harmonics, and fixed source-frame observer projection. FEW's options have relativistic or higher-PN trajectory and amplitude choices, so this receipt makes no numerical FEW equivalence or same-model external-binary claim. No FEW code or output was used.

The generic Kerr/AK/NK orbit, precession, higher-order or self-force flux, all-harmonic amplitude, LISA/TDI response, longer inspiral accuracy, and JAX JIT runtime-domain rejection gates remain open. The older `EMRIKludge` still has only its leading flux component checked against a mathematical reference. Neither class is promoted to the public catalog by this receipt. No deployment or external publication was performed.
