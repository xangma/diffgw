# HM stationary-phase and strain-convention diagnostic

A source-derived convention conversion accounts for most of the approximately
−2.4 rad transition-band primitive phase residual in the
[previous modal diagnostic](imrphenomhm_modal_phase_20260928.md). It does not
remove the later frequency dependence or amplitude error. This receipt adds
an independent [SPA reference](../math/imrphenomhm-spa-conventions.md), a raw
time-domain sign check, and a scratch fixed-conversion comparison on the same
two seen cases. Production waveforms, frozen gates and qualification claims
are unchanged. No new holdout was consumed and no constant was fitted to NR.

## Derivation and source convention

Write $h_{\ell m}=A_{\ell m}q_{\ell m}e^{-im\psi}$ with positive leading
amplitudes and unequal-mass difference. The leading TD phasors in mode order
22, 21, 33, 32, 44, 43 are $(1,i,-i,1,-1,-i)$ in
[Blanchet et al., v3](https://arxiv.org/pdf/0802.1249v3), Eqs. (9.3),
(9.4a,b,d,e,h,i). These are paper equations, not implementation transcriptions.
With the diagnostic carrier $P=\mathcal F_-[\overline h]$, the conjugated
increasing chirp has positive stationary curvature and therefore a $+\pi/4$
SPA contribution. The reference derives this through a damped Gaussian
integral, including the square-root branch, and checks direct TD quadrature.

For $g=\gcd(m,2)$, define the primitive TD and FD quantities

$$J_{\ell m}=v_{\ell m}^{2/g}\overline v_{22}^{m/g},\qquad
T_{\ell m}=u_{\ell m}(f)^{2/g}\overline u_{22}(2f/m)^{m/g},$$

where $v=h/|h|$ and $u=P/|P|$. At leading SPA,

$$T_{\ell m}=\overline J_{\ell m}\exp[i(2-m)\pi/(4g)].$$

The paper's polarization basis in Eqs. (8.2)–(8.3), at observer azimuth
$\pi/2$, has $P=\hat x$, $Q=(0,\cos\iota,-\sin\iota)$.
The spherical basis instead has $e_\theta=Q$, $e_\phi=-P$.
Applying the strain definitions in
[Boyle et al., v2](https://arxiv.org/pdf/1904.04831v2), Appendix C,
Eqs. (C.21)–(C.24), reverses both polarization signs. A symbolic contraction
test checks this identity for an arbitrary symmetric metric perturbation.
This basis conversion predicts the common-sign-reversed coefficient branch
for the SXS-v1 convention. Appendix A.3.1 warns that the older v0 strain
files have the opposite sign; a filename alone is insufficient provenance.

| Convention | Leading $J_{33},J_{44}$ | Leading $T_{33},T_{44}$ |
| --- | --- | --- |
| Printed Blanchet TD coefficients | $-1,-1$ | $e^{3\pi i/4},e^{3\pi i/4}$ |
| Spherical-basis / SXS-v1 strain | $+1,+1$ | $e^{-\pi i/4},e^{-\pi i/4}$ |
| Native orbital-reference London-form low branch | — | $-1,-1$ |

The native row follows from $P_{\ell m}\propto
\exp[-i(m/2)D(2f/m)-i\Delta_{\ell m}]$ and the
[London v4 Eq. (7)](https://arxiv.org/pdf/1708.00404v4) offsets.
Any common change inside the D carrier cancels from $T$. For the SXS-v1
column, the predicted native/NR primitive residual is $e^{-3\pi i/4}$.
The TD check below corroborates this prediction independently of FFTs.

## Raw TD check on len

The scratch script verified both pinned protocol manifests, raw file sizes
and MD5 values, and recorded SHA-256 identities. It loaded local `Strain_N2.h5`
with downloads disabled and required strain type, spin weight −2 and an
identity inertial frame. Both windows were fixed before reading phases:
metadata reference time plus [500,1000) and [1000,1500) $M$, each preceding
the 22 peak. There is no interpolation, FFT, phase unwrapping, alignment or
per-mode fit. The table reports raw circular phase means in radians.

| Case / level | Window, $M$ after reference | $\arg J_{33}$ | $\arg J_{44}$ |
| --- | --- | ---: | ---: |
| 2374 / Lev3 | 500–1000 | −0.073378 | −0.042099 |
| 2374 / Lev3 | 1000–1500 | −0.075338 | −0.042293 |
| 2374 / Lev2 | 500–1000 | −0.073781 | −0.042096 |
| 2374 / Lev2 | 1000–1500 | −0.075433 | −0.042543 |
| 2163 / Lev4 | 500–1000 | −0.071029 | −0.023928 |
| 2163 / Lev4 | 1000–1500 | −0.073246 | −0.023682 |
| 2163 / Lev3 | 500–1000 | −0.070908 | −0.023715 |
| 2163 / Lev3 | 1000–1500 | −0.073239 | −0.023364 |

These phases are near the predicted +1 branch, far from −1; they are not
assumed to reach the exact zero-frequency PN limit. Their circular resultants
exceed 0.999997. All six supported modes were recorded. Every compressed-data
sidecar records the historical overall strain sign change with parent commit
`ef51849550a1d8a5bbdd810c7debf0dd839e86dd`. The sidecar history supports
provenance but is not substituted for the direct phase check or interpreted
as a newly inspected legacy `VersionHist.ver` dataset.

## Fixed FD conversion and remaining error

For the declared SXS-v1 branch, the 22-preserving candidate multiplier is

$$P_{\ell m}\longmapsto
P_{\ell m}\exp[i(m/2-1)3\pi/4].$$

The scratch FD run applied exactly this factor to both frontends' six native
modes, retaining both mappings, both NR resolutions, fixed masks, conditioning
and common-22 alignment from the previous diagnostic. In the table, `L` is
`london_2018`, `R` is `phenomd_runtime`; phases are primitive residual circular
means between their two phase joins, using the higher NR resolution and Torch.

| Case / mode | Before, L / R, rad | After, L / R, rad |
| --- | --- | --- |
| 2374 / 33 | −2.4106 / −2.5630 | −0.0544 / −0.2068 |
| 2374 / 44 | −2.3508 / −2.4571 | +0.0054 / −0.1009 |
| 2163 / 33 | −2.4212 / −2.5557 | −0.0650 / −0.1995 |
| 2163 / 44 | −2.3009 / −2.4076 | +0.0553 / −0.0514 |

Torch/JAX relative complex mode L2 remains at most 6.273e-13. The factor
leaves 22, magnitudes and phase slopes unchanged. Later disagreement remains
substantial: runtime ringdown-window primitive means become +1.415, +1.476,
+2.168 and +2.340 rad in table order. This is not a full-band repair and does
not select a mapping. It leaves the previous amplitude errors intact.

The factor is algebraically a common positive-frequency carrier phase times
an orbital rotation: $e^{-iK}e^{imK/2}$, $K=3\pi/4$. Consequently the
primitive discrepancy alone does **not** prove a production defect.
Nor is a carrier phase automatically a physical polarization rotation:
real polarizations require conjugate negative-frequency factors, whereas
polarization rotation mixes the two real channels. The next gate is a
source-backed signed-mode projection and gauge-transport check. Establish
that conversion before introducing an explicit production convention; then
assess the remaining frequency-dependent phase and amplitude errors separately.
Odd-mode signs still share the common orbital-$\pi$ ambiguity.

## Evidence and replay

The TD run used Python 3.12.3, NumPy 2.5.3 and SXS 2026.0.6. The completed FD
run used the previous CPU FP64 waveform environment: Python 3.12.3, NumPy
2.3.5, SciPy 1.16.3, PyCBC 2.11.0, Torch 2.14.0, JAX/JAXlib 0.11.2 and
SymPy 1.14.0. It ran on len as PID 745791, 2026-09-28 06:46:29–06:46:54 UTC.
The native source identity includes the new reference module; production
waveform files were unchanged. The recorded source snapshot predates the
later source-citation additions to that reference module.

| Artifact | SHA-256 |
| --- | --- |
| `hm_td_convention_diagnostic.py` | `8ba2a66a356a94d303c61854d003e55328eddc9862bcda43637220122beb6939` |
| `td_convention.json` | `fbd71342f7b90918affbd225eb487f2cf34fe3dc33e5ce648c6cea519ce0f4e4` |
| `hm_modal_phase_spa_diagnostic.py` | `cae2e8c2709675a3b66b56cc471a90ac9007fdecd7e150b81ea8a78e779f705e` |
| `fd_convention.json` | `2c65ce7bbf23d6ff4eb286089ea31bfb3fbb508ea31f1974f9e8e46b9a15d0e7` |
| Native Python source tree at FD run | `04a6b74511de040252c98ca5ba15bc47631eadd7479ea81804b440b7ed05ec3e` |

Scripts, outputs and logs reside in `len:/tmp/diffgw-hm-spa-20260928/`.
The `fd-repo/` snapshot there preserves the source used for the FD result.
Frozen input, helper and fixture identities are inherited and rechecked from
the [modal-phase receipt](imrphenomhm_modal_phase_20260928.md#evidence-and-replay).
An initial TD attempt stopped on an unsupported quaternion-array comparison;
converting the frame to an ordinary array fixed that check. Only completed
`td-v2.log` / `td_convention.json` results are reported.

```sh
cd /tmp/diffgw-hm-orbital-20260928/repo
export PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python \
  /tmp/diffgw-hm-spa-20260928/hm_td_convention_diagnostic.py \
  /tmp/diffgw-hm-spa-20260928/td_replay.json
/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  /tmp/diffgw-hm-spa-20260928/hm_modal_phase_spa_diagnostic.py \
  --repo /tmp/diffgw-hm-spa-20260928/fd-repo \
  --output /tmp/diffgw-hm-spa-20260928/fd_replay.json
/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python -m pytest -q \
  tests/test_imrphenomhm_spa_conventions_math_reference.py
```

Scratch curves remain external. No LAL implementation source or fitted oracle
coefficient was used. The maintained tests exercise the derivation and declared
conditional conversion, not whole-waveform physical qualification.

On len, all **36 maintained reference tests passed** in 3.54 s. Strict MkDocs
and both generated waveform-library/coverage consistency checks passed.
