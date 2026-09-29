# NSBH merger-branch classification component gate — 27 September 2026

The [frozen specification](../math/nsbh-branch-classification.md) was
committed as `49054e8` **before** native comparison. Its classification is
the four-way Table I decision in
[Thompson et al. (2020)](https://arxiv.org/pdf/2002.08383v1), also identified
as step 6 of Sec. II.A. Equality $f_{\rm tide}=f_{\rm RD}$ belongs to the
$f_{\rm tide}\ge f_{\rm RD}$ side; only a strictly positive torus ratio has
a torus remnant. The scalar reference uses the printed inequalities directly,
without source or binary output from LAL.

The scalar-only precision and boundary checks passed locally (5/5). On
`len`, the command `JAX_ENABLE_X64=1 PYTHONPATH=src
/home/xangma/repos/diffgw-worktrees/xas-whole-waveform-gate/.venv/bin/python
-m pytest -q -s tests/test_nsbh_branch_reference.py
tests/test_nsbh_torus_reference.py tests/test_nsbh_disruption_reference.py
tests/parity/test_nsbh_parity.py` ran from
`/home/xangma/tmp/diffgw-nsbh-branch-20260927` on Linux x86-64 CPU FP64.
It passed **71/71** in 69.24 s with no skips or expected failures; the log is
`pytest.log` in that directory. The new gate tests all nine exact and
nextafter boundary combinations in each backend, then composes the existing
independent disruption and torus references over 60 physical inputs and three
controlled $f_{\rm RD}/f_{\rm tide}$ ratios each. All four classes occur.
Native Torch and JAX predicates agree **exactly** with the scalar decisions
on every frozen case; the gate has no numerical tolerance.

The neighboring torus gate in the same run reports maximum raw torus-ratio
absolute error $5.329\times10^{-15}$ and clipped error
$2.776\times10^{-16}$ in each backend. The disruption gate reports maximum
relative tidal-frequency error $1.069\times10^{-15}$ (Torch) and
$9.897\times10^{-16}$ (JAX). These gates support the composed branch inputs;
they do not validate a remnant-derived $f_{\rm RD}$.

Environment: Python 3.12.3, Torch 2.14.0+cpu, JAX 0.11.2, NumPy 2.5.3,
mpmath 1.3.0, pytest 9.1.1. This result qualifies the discrete merger-class
selection in the native amplitude path. The remnant/QNM provider,
branch-dependent amplitude fits and windows, phase, full waveform, and
external physical agreement remain separate mandatory gates.
