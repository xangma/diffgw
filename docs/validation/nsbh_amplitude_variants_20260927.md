# NSBH amplitude variant provenance — 27 September 2026

The [source decision and diagnostic limits](../math/nsbh-amplitude-variant-provenance.md)
were frozen in commit `c8520fd` before this diagnostic run. The audited
primary sources are [Thompson et al. (2020), final arXiv v2](https://arxiv.org/pdf/2002.08383v2)
and its cited [Pannarale et al. (2015)](https://arxiv.org/pdf/1509.00512v1).
Thompson v1 and v2 have the same relevant Table I / Eq. (26) / Eq. (29)
forms. Pannarale Eq. (26) corroborates the low/no-torus mixed PN center;
its Eq. (20) also uses a minus ringdown-width window. Thompson explicitly
corrects a factor of two in the latter but does not print a sign correction.
No audited paper equation or erratum uniquely licenses all three native
compatibility variants. This is a **provenance blocker**, not a failed
attempt to fit binary output.

An independent mpmath reference evaluated both variants at 70 and 90 decimal
digits. Every scalar value satisfied the frozen $10^{-30}$ absolute stability
limit. The controlled inputs gave:

| Quantity | Printed paper | Native compatibility | Difference |
| --- | ---: | ---: | ---: |
| Low/no-torus $\widetilde f_0$, $q=3$, $\widetilde f_{\rm RD}=0.09$, $f_{\rm tide}=0.06$, $\epsilon_{\rm ins}=0.8$ | 0.076 | 0.0608 | 20% relative |
| Eq. (26) polynomial at $\Lambda=1$ from below | 0.25 | 1.25 | 1.0 absolute |
| Eq. (29) factor at $x=x_3+d_3/4$ | 0.09684999 | 0.71563001 | 0.61878002 absolute |

Torch and JAX CPU FP64 agreed with the independent **compatibility** scalar
variant at all three points (maximum reported relative error `0.000e+00`,
below the frozen $2\times10^{-13}$ bound). This is a drift check only; the
paper-literal variants do not pass a native equality gate. The diagnostic
does not resolve the additional question of whether Thompson Eq. (26)'s
$\gamma'_1$ is meant to multiply baseline $\gamma_1$.

On `len` (`x86_64` Linux CPU), cwd
`/home/xangma/tmp/diffgw-nsbh-variant-20260927`, Python 3.12.3,
Torch 2.14.0+cpu, JAX 0.11.2, NumPy 2.5.3, mpmath 1.3.0, and pytest 9.1.1,
the focused command
`JAX_ENABLE_X64=1 PYTHONPATH=src python -m pytest -q -s tests/test_nsbh_amplitude_variant_provenance.py`
passed **3/3**. The neighboring run with that file plus
`test_nsbh_joining_frequencies.py`, `test_nsbh_branch_amplitude_fits.py`,
`test_nsbh_branch_reference.py`, `test_nsbh_torus_reference.py`,
`test_nsbh_disruption_reference.py`, `test_nsbh_clean_amplitude.py`,
`test_clean_room_nsbh.py`, and `tests/parity/test_nsbh_parity.py` passed
**107/107** in 78.00 s, with no skips or expected failures; its log is
`nsbh-neighbor.log` in the scratch cwd. Local Ruff, `git diff --check`, and
strict MkDocs build passed.

The native defaults are unchanged. The low/no-torus PN center and Eq. (26)/(29)
paper variants remain unqualified; consequently the next amplitude assembly
gate cannot be declared source-supported. The remnant/QNM provider, amplitude
windows, phase composition, and whole NSBH waveform still require their own
independent qualification.
