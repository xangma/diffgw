# HM intrinsic modes and projection diagnostic — 2026-09-28

This follows the [periodic optimizer audit](imrphenomhm_periodic_optimizer_20260928.md)
on the same seen SXS:BBH:2374 and 2163 fixtures. It separates intrinsic mode
errors from the effects of fitted detector polarization. The calculation uses
the published-calibration/source-PN options, both mapping choices and both
phase conventions with orbital reference phase. It changes no production
waveform, calibration, data conditioning, threshold or historical outcome.

## Mathematical separation

For a stored signed NR mode, the frozen fixture contains Fourier transforms
of its real plus/cross projections at two inclinations. With real
$y_j={}_{-2}Y_{\ell m}(\iota_j,0)$, linearity gives

\[
 B^+_j=y_j U,\quad B^\times_j=y_j V,\qquad
 U=\mathcal F_-[\operatorname{Re}h_{\ell m}],\quad
 V=-\mathcal F_-[\operatorname{Im}h_{\ell m}].
\]

Recover $U,V$ by least squares across views, then form
$\mathcal F_-[h_{\ell m}]=U-iV$ and
$\mathcal F_-[\overline h_{\ell m}]=U+iV$.
This avoids division by a small individual projection coefficient and does not
assume one-sided frequency support. Reprojection, separate-view recovery and
signed-pair symmetry checks test the inversion. The NR carrier compared with
native positive-label modes is
$R_{\ell m}=(-1)^\ell\mathcal F_-[h_{\ell,-m}]$ at positive frequency.
The opposite-frequency contribution and reflection discrepancy are measured
separately, not silently discarded from the reconstruction checks.

This convention follows [García-Quirós et al., Sec. II.1,
Eqs. (2), (4)–(13)](https://arxiv.org/html/2001.10914v2), with the
[existing independent signed-pair adapter](../math/imrphenomhm-signed-projection.md).
That paper's Sec. II.2 also explains the importance of relative mode timing
and the odd-mode sign ambiguity left by quadrupole alignment.

Only the intrinsic $(2,2)$ mode determines a common arrival shift and orbital
angle. The time grid is unchanged. Both orbital branches $\alpha$ and
$\alpha+\pi$ are retained: they leave $(2,2)$ unchanged but flip odd-$m$ modes.
All six modes receive that same transformation. There is no per-mode alignment,
arbitrary complex rotation, amplitude fit or distance adjustment.

For aligned model carrier $P=Ae^{i\phi}$ and NR carrier $R=Be^{i\theta}$,

\[
 |P-R|^2=(A-B)^2+2AB[1-\cos(\phi-\theta)].
\]

Summing each term with the frozen PSD weight separates amplitude and phase
contributions exactly. The amplitude term is independent of time, orbital
phase and detector polarization, and is a lower bound on the complex error
under phase-only transformations. Modal norm ratios and amplitude errors are
therefore useful even when the orbital branch is ambiguous. Wrapped-phase
statistics require nonzero amplitudes; the decomposition itself remains valid
at zeros. Modal powers do not add to a detector's coherent power.

Projection diagnostics compare the old full-signal optimum with its
polarization reset to the fixed source value, holding the orbital/time alignment
fixed. A separate quadrupole-aligned projection uses the source polarization
throughout. Resetting one parameter is an explanatory ablation, not another
optimized match or a replacement qualification gate.

## Results

All eight case/mapping/convention combinations completed on len. Intrinsic
$(2,2)$ alignment gives matches 0.999248 (2374) and 0.999386 (2163), with
norm ratios 1.006529 and 1.010398. Neither arrival-time optimum lies on the
search boundary. These are diagnostic alignments, not newly frozen gates.

### Amplitude discrepancy survives every phase choice

The following numbers are invariant under orbital branch, arrival shift and
SPA/legacy choice. $E_A=\|\,|P|-|R|\,\|/\|R\|$ uses the frozen PSD.

| Case | Mode | London norm ratio | Runtime norm ratio | London $E_A$ | Runtime $E_A$ |
| --- | --- | ---: | ---: | ---: | ---: |
| 2374 | 22 | 1.0065 | 1.0065 | 0.0091 | 0.0091 |
| 2374 | 21 | 0.5294 | 0.5549 | 0.5193 | 0.5104 |
| 2374 | 33 | 1.0818 | 1.0881 | 0.0866 | 0.0933 |
| 2374 | 32 | 1.1961 | 1.2103 | 0.3764 | 0.3761 |
| 2374 | 44 | 1.2082 | 1.2127 | 0.2312 | 0.2344 |
| 2374 | 43 | 1.1896 | 1.1917 | 0.4185 | 0.4186 |
| 2163 | 22 | 1.0104 | 1.0104 | 0.0138 | 0.0138 |
| 2163 | 21 | 0.9339 | 0.9437 | 0.1557 | 0.1499 |
| 2163 | 33 | 1.0795 | 1.0830 | 0.0928 | 0.0964 |
| 2163 | 32 | 1.2964 | 1.3026 | 0.6374 | 0.6385 |
| 2163 | 44 | 1.3613 | 1.3639 | 0.4303 | 0.4318 |
| 2163 | 43 | 0.7089 | 0.7102 | 0.4339 | 0.4335 |

Thus a nearly correct 22 amplitude coexists with substantial higher-mode
amplitude errors. An amplitude-preserving convention change cannot remove
these errors. Changing mapping transitions has little effect on these integrated
amplitude metrics; it is not a sufficient amplitude remedy here.

These are per-mode errors, not contributions to total detector mismatch.
The six intrinsic NR power fractions for $(22,21,33,32,44,43)$ are respectively
$(89.0223,1.4415,8.1389,0.1628,1.2058,0.0287)$ percent for 2374 and
$(94.2753,0.1867,4.6985,0.2251,0.5898,0.0247)$ percent for 2163.
Weak-mode relative errors must be interpreted with these powers and the
observer-dependent coherent projection.

### Phase errors also remain

For runtime/SPA, the even modes avoid the odd-branch ambiguity entirely.
After the common 22 alignment, their phase-error powers divided by their own
NR modal power are:

| Case | 22 | 32 | 44 |
| --- | ---: | ---: | ---: |
| 2374 | 0.00147 | 0.78262 | 0.50639 |
| 2163 | 0.00116 | 2.31783 | 1.20227 |

These are the phase terms in the exact decomposition above, not mismatch
percentages. They can exceed one. Both odd-mode branches and both conventions
remain in the scratch result; no branch is chosen independently for a mode.
For example, runtime/SPA 33 has fixed-alignment signed matches
$\pm0.95031$ (2374) and $\pm0.91575$ (2163), whereas 43 has
$\pm0.19442$ and $\pm0.02929$.

In 2163, 96.39% of runtime 32 amplitude-error power lies in 200–400 Hz;
its amplitude-product-weighted phase concentration in that band is only
0.4110. For 43 the corresponding numbers are 71.61% and 0.5234.
Concentration is the modulus of the weighted mean phase phasor; low values
show frequency-dependent phase variation that a constant mode rotation cannot
remove. This is separate from the coherent residual bands in the preceding
audit. Concentration is undefined where candidate/reference amplitude-product
weight vanishes; such bands retain amplitude error but have no phase statistic.

### Projection explains part of the optimized norm excess

For the second view (inclination 1.2, source polarization 0.4 rad), runtime/SPA:

| Case | Alignment | Norm ratio | Signed overlap |
| --- | --- | ---: | ---: |
| 2374 | Previous full optimum | 1.143846 | 0.983052 |
| 2374 | Same time/orbital angle, source polarization | 0.984610 | 0.914963 |
| 2163 | Previous full optimum | 1.292795 | 0.981391 |
| 2163 | Same time/orbital angle, source polarization | 1.082548 | 0.837640 |

The 1.2928 detector norm is therefore not a measurement of intrinsic 22
amplitude error. Polarization and coherent interference contribute substantially.
Resetting polarization does not solve the waveform: overlap deteriorates.
With the independent 22 anchor and source polarization, the two common
orbital branches give second-view overlaps (0.972844, 0.675292) for 2374
and (0.814676, 0.968081) for 2163. Even the larger of each diagnostic pair
falls below the historical 0.98 model-match floor. These fixed-orientation
ablations do not replace the historically optimized qualification protocol.

### Recovery controls

Across both cases and all signed modes, reprojection relative L2 is at most
$1.63\times10^{-15}$ and the separate-view reconstruction error at most
$5.76\times10^{-15}$. Maximum absolute inverse gain is 15.455; the scalar
least-squares condition number is trivially one and is not by itself a stability
claim. An independent raw 2374 $(2,-2)$ FFT with the identical frozen grid,
taper and interpolation agrees to $2.88\times10^{-16}$ for positive frequency
and $5.72\times10^{-12}$ for its weak conjugate negative-frequency contribution.
This checks recovery, not sensitivity to the chosen conditioning.

The largest modeled-mode reflection residual power fraction is
$4.77\times10^{-8}$ for 2374 and $3.09\times10^{-10}$ for 2163.
Maximum wrong-helicity power fractions are $9.09\times10^{-5}$ and
$2.46\times10^{-7}$. Reprojecting recovered carriers with the native one-sided
pair approximation has relative error power at most $9.13\times10^{-5}$
and $2.48\times10^{-7}$, respectively. These small convention/one-sidedness
residuals do not explain the large intrinsic errors above. They do not test
all possible NR systematic errors.

Torch/JAX intrinsic-mode disagreement is below $6.28\times10^{-13}$ relative
L2. All exact amplitude/phase decomposition, frozen evidence identity, prior
optimum replay and coherent reconstruction assertions passed.

## Interpretation against the source model

London's [PhenomHM paper, Eqs. (3)–(9) and Results on PDF p. 3](https://arxiv.org/pdf/1708.00404v4)
constructs higher modes by approximate quadrupole mapping. It reports weaker
$(3,2)$ and $(4,3)$ agreement associated with spherical–spheroidal mixing and
only approximate higher-mode amplitudes. These establish plausible model
limitations, not a causal explanation for an individual residual in this audit.
Identifying mixing specifically requires a source-backed spherical/spheroidal
comparison; an error in $(3,2)$ alone is insufficient evidence.

The diagnostic identifies both intrinsic amplitude/phase discrepancies and a
projection contribution to the optimized norm. It does not establish a new
frontend transcription defect or uniquely identify mode mixing. The independent
complete-waveform reference remains the implementation check; it does not
make the approximate source model exact against NR.

The next bounded source-model audit should isolate the higher-mode amplitude
construction, starting with the strong 2374 21 deficit and 2163 44 excess.
Evaluate the published Eq. (4) factors and the paper's explicitly stated
alternative amplitude factor independently, component by component, with the
existing carrier held fixed; check conditioning sensitivity before interpreting
the NR residual as model error. Treat any alternative as a separately declared
diagnostic variant, not a fitted correction. The 32/43 frequency-dependent
phase residuals need a separate source-backed mixing/phase investigation.
Neither adjustment can be claimed as qualified using these already seen cases.
All historical gates, options and qualification statuses are preserved.

## Execution and reproducibility

The final run used source revision `8521a9b085c34228d66d4754d9db3ea6214a28e5`,
whose native source hashes equal those in the preceding periodic audit.
It ran on len, CPU FP64, PID 3563754, from 13:41:54 to 13:42:30 UTC on
2026-09-28 and completed with `qualification_promoted=false`. No job remains
active. Working directory: `/tmp/diffgw-hm-intrinsic-projection-20260928`.
The archive is in `repo/`; scratch outputs are `intrinsic-final.json`,
`intrinsic-final.raw.npz` and `run-final.log`.

The backend environment was `/tmp/diffgw-waveform-library-1315f6e/venv-latest`:
Python 3.12.3, NumPy 2.3.5, SciPy 1.16.3, PyCBC 2.11.0, Torch 2.14.0,
JAX/jaxlib 0.11.2 and SymPy 1.14.0. The raw-reader subprocess used
`/tmp/diffgw-hm-sxs-2265-20260927/.venv/bin/python`, with SXS 2026.0.6,
NumPy 2.5.3 and SciPy 1.18.1. No packages were changed for this diagnostic;
the two existing environments separate raw NR decoding from native evaluation.

The first run stopped on missing `sxs` in the backend environment; its
`intrinsic.json` and `run.log` are retained. The separate-reader retry completed.
The final run additionally corrects alignment descriptions in diagnostic output
and asserts that both prior overlap and norm replay exactly. Earlier outputs
are not the cited final evidence.

Replay with all three unchanged helpers from the preceding audit beside the
scratch driver and a fresh output path:

```sh
cd /tmp/diffgw-hm-intrinsic-projection-20260928
export PYTHONPATH="$PWD/repo/src" JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/tmp/diffgw-waveform-library-1315f6e/venv-latest/bin/python \
  hm_intrinsic_diagnostic.py --repo "$PWD/repo" \
  --source-revision 8521a9b --output /tmp/new-hm-intrinsic.json
```

| Artifact | SHA-256 |
| --- | --- |
| Final JSON | `12114e33b00c2f68b18002bd0710db5ab4b3789652d8a484c7263ca340aee85f` |
| Scratch driver | `cd63e2df35749279898105c77072b5773c4c21511c1638f88e2c7e7c13519fbc` |
| Prior periodic JSON | `cbd3097dbb58f443050756d31d73870a70afb831f120f358c341b054aabbe2fd` |

Helper hashes, manifests, raw files, fixtures, PSD, sidecars, preparation and
historical metrics are verified against the frozen evidence chain before reuse.
Source hashes are checked before and after the calculation. The historical NR
resolution screen is reused, not regenerated. Scratch scripts and complete
JSON outputs remain external temporary artifacts; this maintained receipt
preserves the method, numerical findings and checksums, not permanent archival
availability of `/tmp` files.

Documentation validation on len: generated library and coverage checks and
strict MkDocs build passed. No numerical regression suite was rerun for this
documentation-only change.
