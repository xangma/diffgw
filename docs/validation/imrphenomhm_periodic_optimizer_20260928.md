# HM periodic optimizer and common-alignment residual audit — 2026-09-28

This diagnostic follows the [native SPA/SXS rescore](imrphenomhm_spa_native_20260928.md)
on already examined SXS:BBH:2374 and 2163. It distinguishes search failures
from waveform discrepancies. It does not change a waveform, calibration,
PSD, conditioning, data selection, threshold or historical qualification result.
The source revision is `40bcf50463f9ef594b8822017c7e0c9007829dc2`.

## Optimizer mathematics

The existing real detector inner product is

\[
(a,b)=4\operatorname{Re}\sum_f a(f)^*b(f)/S_n(f),
\]

with the frozen 1 Hz frequency spacing. At orbital angle $\alpha$, let
$P=\sum_{\ell m}p_{\ell m}e^{-im\alpha}$ and
$X=\sum_{\ell m}x_{\ell m}e^{-im\alpha}$. Physical polarization uses
$a=(\cos2\psi,\sin2\psi)$ and $h=a_1P+a_2X$.
For a common arrival shift $\tau$, define

\[
G_{ij}=(u_i,u_j),\quad
b_i(\tau)=(r,u_i e^{-2\pi if\tau}),\quad R=(r,r),
\qquad u=(P,X).
\]

Cauchy–Schwarz in the positive-definite Gram metric gives

\[
\max_\psi M=\sqrt{b^T G^{-1}b/R},\qquad
 a=\frac{G^{-1}b}{\|G^{-1}b\|_2},\qquad
 \psi=\tfrac12\operatorname{atan2}(a_2,a_1)\pmod\pi.
\]

Euclidean normalization of $a$ is essential for the physical candidate norm;
whitening or arbitrary amplitude normalization would change the distance.
There is no extra free complex phase. Gram rank and conditioning must be
checked; a singular Gram matrix can make the physical norm nonunique even
when the maximum match is defined.

The audit eliminates polarization analytically, scans orbital phase
periodically, and retains the original discrete arrival shifts
$k/8192$ seconds for $k=-819,\ldots,819$. It introduces no continuous-time
refinement. Grid convergence is numerical evidence, not a proof of global
optimality. The original four-seed results remain frozen.

## Residual accounting

Every native mode receives the same winning $(\alpha,\psi,\tau)$ from the
complete detector waveform. The corresponding NR component is the sum of
its signed $(\ell,+m)$ and $(\ell,-m)$ pair, projected with the fixed source
polarization. Modes and frequency bands receive no further alignment or fit.

For each supported pair, $e_j=h_j-r_j$; the omitted NR contribution is
$e_{\rm omitted}=-r_{\rm omitted}$. Their sum reconstructs $e=h-r$.
Individual residual powers do not add because modes interfere. The signed
contributions

\[
C_j=(e_j,e)/R,\qquad \sum_j C_j=(e,e)/R
\]

provide additive accounting and may be negative. Disjoint frequency bands
are [30,60), [60,100), [100,200), [200,400), and [400,1025) Hz. Their
residual powers add directly. A band's contribution is normalized by the
full reference power, not by that band's power.

## Results and limitations

The largest improvement over any previous best match is
$9.39\times10^{-12}$. All 16 match/norm outcomes are unchanged. The new search
agrees across all six grid/origin passes within $5.56\times10^{-16}$;
all refinements succeed and no winning NR arrival shift is on a search boundary.
This resolves the interpretation of the three seed-spread failures: poorer
seeds had stalled, but the previous best scores were already accurate.

| Historical seed-spread failure | Audited explanation |
| --- | --- |
| 2374, London/legacy, view 0 | Seeds 17/67 stopped at the orbital boundary. Periodic wrapping recovers the existing best near $\alpha=6.21433$. |
| 2163, London/SPA, view 0 | Seed 41 chose a poorer polarization/time branch. Analytic elimination at its orbital angle selects time bin 55 instead of 56; orbital refinement recovers the existing best. |
| 2163, runtime/SPA, view 1 | Seeds 23/67 stopped near $\psi=\pi$. Periodic polarization and orbital refinement recover the existing best; the norm remains 1.292795. |

The following table applies only the unchanged match floor 0.98 and norm
interval [0.85,1.15]. Ten of sixteen comparisons satisfy those two conditions.
It excludes the historical seed-consistency gate and does not replace its
frozen outcomes. Both phase conventions use the orbital reference-phase option;
these rows must not be conflated with the original legacy-reference-phase NR run.

| SXS | Mapping / convention | View | Match | Norm | Match/norm gates |
| --- | --- | --- | --- | --- | --- |
| 2374 | London / legacy | 0 | 0.988641 | 0.937321 | Pass |
| 2374 | London / legacy | 1 | 0.971789 | 0.811100 | Fail |
| 2374 | London / SPA | 0 | 0.988881 | 0.956175 | Pass |
| 2374 | London / SPA | 1 | 0.975269 | 1.112124 | Fail |
| 2374 | runtime / legacy | 0 | 0.995016 | 0.962849 | Pass |
| 2374 | runtime / legacy | 1 | 0.979583 | 0.984988 | Fail |
| 2374 | runtime / SPA | 0 | 0.995329 | 0.984803 | Pass |
| 2374 | runtime / SPA | 1 | 0.983052 | 1.143846 | Pass |
| 2163 | London / legacy | 0 | 0.990156 | 0.976356 | Pass |
| 2163 | London / legacy | 1 | 0.977392 | 0.824196 | Fail |
| 2163 | London / SPA | 0 | 0.990120 | 0.972344 | Pass |
| 2163 | London / SPA | 1 | 0.974579 | 0.941364 | Fail |
| 2163 | runtime / legacy | 0 | 0.993991 | 0.976669 | Pass |
| 2163 | runtime / legacy | 1 | 0.984505 | 0.933001 | Pass |
| 2163 | runtime / SPA | 0 | 0.994074 | 0.999716 | Pass |
| 2163 | runtime / SPA | 1 | 0.981391 | 1.292795 | Fail |

The London/SPA second-view failures have their largest positive additive
residual contributions in the $(2,2)$, $(3,3)$ and $(4,4)$ pairs. Their
reference-normalized total residual powers are 0.067581 (2374) and 0.051299
(2163). The runtime/SPA 2163 second view is a useful contrasting case: its
match passes but its norm fails. At its common optimum, total residual power
is 0.133844 times the reference power:

| Signed mode pair | Projected norm ratio | Fixed-alignment match | Signed share of total residual power |
| --- | --- | --- | --- |
| (2,2) | 1.209701 | 0.993830 | 48.10% |
| (2,1) | 1.267622 | 0.543516 | 7.27% |
| (3,3) | 1.296585 | 0.981237 | 11.83% |
| (3,2) | 1.845510 | 0.044276 | 16.85% |
| (4,4) | 1.632973 | 0.847387 | 11.08% |
| (4,3) | 1.005062 | -0.043598 | 1.76% |
| Omitted NR modes | — | — | 3.11% |

| Frequency band (Hz, upper exclusive) | Share of total residual power |
| --- | --- |
| 30–60 | 15.84% |
| 60–100 | 12.37% |
| 100–200 | 53.13% |
| 200–400 | 17.94% |
| 400–1025 | 0.72% |

These numbers locate discrepancies in the coherently projected signal; they
do not identify unique intrinsic-mode defects. The fitted polarization differs
from the fixed NR source polarization, so projected norm ratios are not intrinsic
amplitude ratios. Interference also matters: the 2374 London/SPA $(2,1)$
contribution is negative (−4.34%). Independently fitting each mode, rescaling
amplitudes, or treating these percentages as causal ablations would misrepresent
the evidence.

The next discriminating investigation is to separate projection effects from
intrinsic mode amplitude and frequency-dependent phase at a common source
orientation, following the published equations and the independent reference.
The $(3,2)$ discrepancy and the 100–200 Hz band merit attention in the runtime
2163 case, alongside $(2,2)$, $(3,3)$ and $(4,4)$ in the London cases.
More optimizer seeds are not the immediate remedy. Any resulting production
change needs source-backed mathematics and independent frontend checks, followed
by a frozen variant/domain and suitable untouched physical validation data.
No qualification claim, threshold, historical failure or production scorer is
changed by this diagnostic.

## Execution and reproducibility

The completed audit ran on `len`, CPU FP64, from 12:22:30 to 12:23:56 UTC
on 2026-09-28, Python PID 1404552 (wrapper PID 1404550), returning zero.
The working directory was `/tmp/diffgw-hm-periodic-audit-20260928/repo`, an
archive of the source revision above. The parent directory retains the scratch
modules, `audit.log`, `audit.exit` and `audit.json`. The JSON reports
`status="completed"` and `qualification_promoted=false`; these temporary
artifacts are not a maintained production optimizer or permanent evidence store.

Use the same environment as the [native rescore](imrphenomhm_spa_native_20260928.md#execution-and-replay):
`/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin` on `PATH`.
The audit verifies the prior receipt hash, helper and independent projection,
raw-file pins, preparation identity, fixtures, sidecars, PSD and historical
metrics. It reuses the historical NR-resolution screen only after those
identity checks; it does not rerun that screen. Source hashes are checked
before and after execution.

```sh
cd /tmp/diffgw-hm-periodic-audit-20260928/repo
export PATH=/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin:$PATH
export PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python ../hm_periodic_audit_driver.py --repo "$PWD" \
  --source-revision 40bcf50463f9ef594b8822017c7e0c9007829dc2 \
  --output /tmp/new-hm-periodic-audit.json
```

Orbital grids contain 256, 512 and 1024 points, each with offsets of zero and
half a cell. Every sampled envelope maximum receives wrapped scalar refinement.
Competitive time-branch maxima within 0.002 of the best score on the finest
grid also receive refinement; the declared cap of 256 excludes no eligible
refinement in this run. This is a converged diagnostic search, not a certified
bound over continuous angles.

Synthetic checks recover two planted multiharmonic signals near both angular
boundaries, verify the analytic polarization maximum against 8192 sampled
polarizations, compare FFT correlations with direct inner products, check
smooth fixed-time derivatives, and explicitly expose norm ambiguity at Gram
rank one. Every actual NR optimum has rank two; the largest condition number
is 2.552. Maximum discrepancies over the 16 actual comparisons are:

| Check | Maximum error |
| --- | --- |
| Analytic score versus direct frozen score | $7.78\times10^{-16}$ absolute |
| Native projection at the common optimum | $2.37\times10^{-13}$ relative L2 |
| Torch/JAX intrinsic-mode parity | $6.28\times10^{-13}$ relative L2 |
| Additive residual accounting | $2.17\times10^{-15}$ absolute |

The optimization is performed once per comparison using Torch modes; both
native frontends are evaluated at that same optimum. Every historical seed is
replayed, and analytic polarization/time optimization at its orbital angle is
checked not to reduce its score. The best historical score must also be recovered
within $10^{-9}$. Mode reconstruction and disjoint-band accounting are checked
separately, without extra mode alignment or amplitude fitting.

| Artifact | SHA-256 |
| --- | --- |
| Source archive | `4f50fed81117aded783138d0372f4efba014ea6cca2e2f9f495e2811681d2825` |
| `hm_periodic_audit_driver.py` | `4d3c69d0e34931bcfc72b4eb7795db06c4a159197b181f907e2a2870174f5ffb` |
| `hm_periodic_optimizer.py` | `f1aac1e43ab43ea6e4b179bb7db260a556beef52d21e3251cc6bed0f24779c74` |
| Prior helper `hm_spa_native_rescore.py` | `0426a424e853e71ccd6f6111279c6647954e32f336f1dc103b0009e9859c58bb` |
| Prior `rescore.json` | `f3fad6cf7693ac26d4645a2f2305b9382635e9ed364ba8bbe8ac48ca8e14b659` |
| `audit.json` | `cbd3097dbb58f443050756d31d73870a70afb831f120f358c341b054aabbe2fd` |

The source revision's [nine CI jobs](https://github.com/xangma/diffgw/actions/runs/36405563430)
passed, including mathematical-reference and binary-oracle checks. Those
implementation checks and this diagnostic audit remain distinct from physical
qualification.

The follow-up [intrinsic/projection diagnostic](imrphenomhm_intrinsic_projection_20260928.md)
separates intrinsic amplitude/phase discrepancies from polarization effects,
using these same frozen cases and optima.
