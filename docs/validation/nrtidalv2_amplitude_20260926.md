# NRTidalv2 amplitude and merger-taper diagnostic — 26 September 2026

**Result: Eq. (24) component math qualified; explicit `paper_2019` D
composition qualified for this amplitude/taper slice; historical default
binary compatibility retained, with amplitude provenance unresolved.**
This receipt covers only the nonspinning tidal
amplitude and merger taper specified in
[nrtidalv2-amplitude-taper/v1](../math/nrtidalv2-amplitude-taper.md). The
separate [nonspinning phase receipt](nrtidalv2_phase_20260926.md) remains
intact. No coefficient was fit to the binary, and no LAL implementation source
was inspected.

[Dietrich et al. (2019), Table II and accompanying text](https://arxiv.org/pdf/1905.06011v2)
specify that `IMRPhenomD_NRTidalv2` adds only the tidal phase to the BBH
carrier; the Eq. (24) tidal-amplitude correction belongs to
`IMRPhenomPv2_NRTidalv2`. [Colleoni et al. (2025), Sec. II A](https://arxiv.org/pdf/2311.15978v2)
independently confirms this older-model distinction. Both Torch/JAX D frontends
now expose `amplitude_variant="paper_2019"`, which omits tidal amplitude. Their
default `binary_compatible` variant preserves the earlier native
$M_{\rm sec}^2/(2D_{\rm sec})$ correction and the frozen pinned-binary output.
The origin of its $1/2$ normalization remains unresolved; it is **not** a
published Eq. (24) coefficient or a claim that the default implements the
paper's D composition. The published Eq. (24) component remains separately
checked by Torch/JAX/SymPy and is used by neither D variant.

The earlier [Kawaguchi et al. (2018), Eq. (2.25)](https://arxiv.org/pdf/1802.06518)
uses $\sqrt{5\pi\eta/24}M^2/D_{\rm eff}$ for its absolute additive tidal
amplitude. [Dietrich et al. (2019), Eq. (22)](https://arxiv.org/pdf/1905.06011v2)
explicitly substitute $D_L$ for $D_{\rm eff}$, and Eqs. (23)–(24) retain the
same $\eta$ scaling. The independent SymPy reference now proves that the
Kawaguchi expression equals the numerator of Dietrich Eq. (24), including
its 7.890-power term. The historical native/published scalar ratio is
$[2\sqrt{5\pi\eta/24}]^{-1}$, giving 1.236077446 for the equal-mass case
and 1.248886881 for the unequal-mass case; both Torch and JAX are tested
against this distinction. With inclination fixed at 0.4 rad, a single
projection factor cannot explain both ratios. This is a **paper versus
pinned-binary contract difference**, not evidence that $1/2$ is a published
coefficient. [The first 2019 preprint, Table II](https://arxiv.org/pdf/1905.06011v1)
already marks D_NRTidalv2 tidal amplitude absent. The pinned binary version,
frozen cases, and measured output disagreement appear below.

The manifest's `verified_declared_cases` status and frozen metrics apply to
the **default binary-compatible contract only**. The explicit paper-D option
passes its math-composition tests but remains unqualified against the pinned
LALSuite 7.26.15 D binary at nonzero tide. The strict oracle catalog and
holdout inventory and limits are unchanged.

The unchanged `scripts.run_oracle_ci` gate passed on `len` in the same cwd
with the checksum-verified `SEOBNRv4ROM_v3.0.hdf5`, LALSuite distribution
7.26.15 and PyCBC 2.11.0: **catalog 19/19, holdout 14/14**. The D_NRTidalv2
catalog row was `PASSED` with plus complex relative $L_2=3.3468418936\times
10^{-9}$ and maximum projection $L_2=2.9618710589\times10^{-7}$; its
holdout row was `PASSED` with plus $L_2=5.6726113326\times10^{-9}$ and
maximum projection $L_2=3.0914191861\times10^{-7}$. The log is
`nrtidal_dual_oracle_ci_pycbc.log`, with the unmodified reports
`parity_results.json` and `parity_holdout_results.json`, in the remote cwd.
The first attempt with a venv lacking PyCBC could not evaluate the required
optimized-match metric and marked all ordinary strict cases failed; its
`nrtidal_dual_oracle_ci.log` is environmental evidence, not a model result.

## 27 September provenance follow-up

The independent Kawaguchi/Dietrich symbolic identity and the two-mass-ratio
Torch/JAX normalization distinctions added after the initial receipt passed
on `len` in `/home/xangma/tmp/diffgw-nrtidal-paper-d-20260926`: **67 passed,
18 strict xfailed** in `nrtidal_provenance_focus.log`. The focused command is
the four-file command below with the same pinned LALSuite 7.26.15 environment.
The unchanged `python -m scripts.run_oracle_ci` strict gate, run with PyCBC
2.11.0 and the verified ROM data in the same cwd, passed **catalog 19/19,
holdout 14/14**; see `nrtidal_provenance_oracle_ci.log`. No production waveform
code, frozen oracle inventory, threshold, or expected-failure status changed
in this follow-up. The scientific release distinction remains: the published
2019 D composition is source-scoped, while the default is binary-verified
with unresolved amplitude provenance.

On `len`, the dual-contract focused CPU FP64 run in
`/home/xangma/tmp/diffgw-nrtidal-paper-d-20260926` returned **62 passed,
18 strict xfailed**. The exact command and output are in
`nrtidal_dual_focus.log` there. It covered the scalar paper reference,
Torch/JAX component and composition checks, both pinned-binary comparisons,
and Torch reconstruction:

```sh
PYTHONPATH=src:tests:/home/xangma/diffgw-pr45-318b37f7/.oracle-deps \
DIFFGW_REQUIRE_LAL_BINARY=1 JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -s \
tests/test_nrtidalv2_amplitude_math_reference.py \
tests/test_nrtidalv2_amplitude_math_reference_native.py \
tests/test_nrtidalv2_amplitude_lal_binary.py \
tests/test_nrtidal_reconstruction.py
```

| Explicit paper-D comparison with pinned D binary | Frozen maximum | Observed maximum | Result |
|---|---:|---:|---|
| Equal mass, through merger | absolute $5\times10^{-5}$ | $0.355275595$ | strict xfail |
| Unequal and exchanged, through merger | absolute $5\times10^{-5}$ | $0.245061191$ | strict xfail |
| Equal mass, in taper | absolute $5\times10^{-5}$ | $0.378202841$ | strict xfail |
| Unequal and exchanged, in taper | absolute $5\times10^{-5}$ | $0.261641146$ | strict xfail |
| Zero tide | absolute $5\times10^{-5}$ | $\le2.3\times10^{-16}$ | pass |

These paper-D versus pinned-binary discrepancies are retained as strict
expected failures in a diagnostic test. They are not substituted into the
frozen catalog or holdout cases. The default's pre-taper native/LAL amplitude
ratio passes the $5\times10^{-5}$ limit, with maximum $1.84458477\times10^{-7}$.
Inside the taper, the unchanged $5\times10^{-5}$ limit still fails by
$0.000153757344$ (equal mass) and $0.000189164604$ (unequal/exchanged),
under the pre-existing strict expected-failure markers. This small boundary
difference is distinct from the paper-D composition discrepancy above.

## Earlier Eq. (24)-in-D diagnostic

On the earlier Eq. (24)-in-D branch, Torch and JAX used the published
$\sqrt{5\pi\eta/24}\,M^2/D_L$ scale. The historical native $M^2/(2D_L)$ scale
gave a mass-ratio-dependent relative error of 0.2361–0.2489 against Eq. (24);
it now remains only in the binary-compatible default. The Eq. (24) scale is
supported independently by the physical IMRPhenomD
Newtonian carrier: $\sqrt{5\eta/24}M^{5/6}/(\pi^{2/3}D_Lf^{7/6})
=\sqrt{5\pi\eta/24}M^2x^{-7/4}/D_L$. The identity follows from
[Cutler and Flanagan, Eqs. (62)–(63)](https://arxiv.org/pdf/gr-qc/9402014v1)
and $x=(\pi Mf)^{2/3}$. We changed both backends and the paper-math tests;
no new LAL-derived normalization was introduced. On `len`, that focused test
command covering the scalar, native waveform, pinned binary and Torch
reconstruction returned **54 passed, 18 strict xfailed**. The 18 expected
failures are the nonzero-tide paper-versus-LAL and corrected-native-versus-LAL
quotients before/inside the taper. The frozen limits did not change.

That run did **not** establish why the pinned LAL binary outputs matched the earlier
native normalization, and the approximately 25 ppm effective merger-boundary
offset reported below remained unresolved. The LAL binary gate for that
abandoned Eq. (24)-in-D composition was red.
The later **pre-correction baseline** table is retained to show how the
earlier native normalization compared with Eq. (24) and the binary.

Equation (24) is an absolute **additive** tidal amplitude, not the complete
BNS amplitude and not a relative factor. Equation (25) forms
$(\widetilde A_{\rm BBH}+\widetilde A_T)W$. The earlier D frontends implemented this as
$h_{\rm BBH}(1+\widetilde A_T/\widetilde A_{\rm BBH})W$, using the same
physical BBH carrier amplitude in the denominator. This division only
applies the additive correction to the complex waveform; it does not
multiply by a second carrier normalization.

| Earlier Eq. (24)-in-D comparison | Frozen maximum | Observed maximum | Result |
|---|---:|---:|---|
| Torch/JAX scalar signed amplitude / Eq. (24) | relative $10^{-11}$ | at most $10^{-11}$ | pass |
| Torch/JAX untapered $h_+$ modulus quotient / paper | absolute $10^{-10}$ | at most $10^{-10}$ | pass |
| LAL versus paper; native versus LAL, through merger, equal mass | absolute $5\times10^{-5}$ | $0.0678536536$ | fail |
| Same, unequal and exchanged | absolute $5\times10^{-5}$ | $0.0488374025$ | fail |
| LAL versus paper; native versus LAL, after merger, equal mass | absolute $5\times10^{-5}$ | $0.0722370285$ | fail |
| Same, unequal and exchanged | absolute $5\times10^{-5}$ | $0.0521550463$ | fail |

The earlier focused run used host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-amplitude-eq24-20260926`, and
`amplitude_eq24_focus_final.log` there. Its command was:

```sh
PYTHONPATH=src:tests:/home/xangma/diffgw-pr45-318b37f7/.oracle-deps \
DIFFGW_REQUIRE_LAL_BINARY=1 JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -s \
tests/test_nrtidalv2_amplitude_math_reference.py \
tests/test_nrtidalv2_amplitude_math_reference_native.py \
tests/test_nrtidalv2_amplitude_lal_binary.py tests/test_nrtidal_reconstruction.py
```

The broader `len` run in the same cwd used `amplitude_eq24_broad.log` and
returned **94 passed, 1 skipped** in 114.65 s. It covered
`tests/test_nrtidal.py`, `tests/test_nrtidal_reference.py`,
`tests/parity/test_nrtidal_parity.py`,
`tests/parity/test_waveforms_dispatch_parity.py`, and
`tests/test_nrtidalv2_phase_math_reference.py` under the same environment.

## Pre-correction baseline

The independent SymPy/mpmath reference transcribes the full absolute
normalization of [Dietrich et al. (2019), Eq. (24)](https://arxiv.org/pdf/1905.06011v2),
with its $\sqrt{5\pi\eta/24}$ prefactor, and its Eq. (25) Planck taper.
This normalization is internally consistent with Eq. (23): its leading
$-27x^5/16$ term times $\sqrt{160\pi\eta/27}\,x^{-7/4}$ gives
$-9\sqrt{5\pi\eta/24}\,x^{13/4}$ exactly. The source therefore does
not support replacing Eq. (24)'s prefactor with the native $1/2$ factor.
The merger fit follows [Dietrich et al. (2018), Eq. (11)](https://arxiv.org/pdf/1804.02235).
The frozen systems, frequencies and limits are in the specification and
tests. They were written before the first pinned-binary run and were not
relaxed after its failure.

| Comparison | Frozen maximum | Observed maximum | Result |
|---|---:|---:|---|
| Torch/JAX scalar signed amplitude / Eq. (24), equal mass | relative $10^{-11}$ | $0.23607725316$ | fail |
| Torch/JAX scalar signed amplitude / Eq. (24), unequal and exchanged | relative $10^{-11}$ | $0.24888677449$ | fail |
| Torch/JAX untapered $h_+$ modulus quotient / paper, equal mass | absolute $10^{-10}$ | $0.01553696317$ | fail |
| Torch/JAX untapered $h_+$ modulus quotient / paper, unequal and exchanged | absolute $10^{-10}$ | $0.01795564024$ | fail |
| Pinned LAL BNS/BBH modulus quotient / paper, through merger, equal mass | absolute $5\times10^{-5}$ | $0.0678536536$ | fail |
| Pinned LAL BNS/BBH modulus quotient / paper, through merger, unequal and exchanged | absolute $5\times10^{-5}$ | $0.0488374025$ | fail |
| Pinned LAL BNS/BBH modulus quotient / paper, after merger, equal mass | absolute $5\times10^{-5}$ | $0.0722370285$ | fail |
| Pinned LAL BNS/BBH modulus quotient / paper, after merger, unequal and exchanged | absolute $5\times10^{-5}$ | $0.0521550463$ | fail |
| Torch/JAX vs pinned LAL quotient through merger | absolute $5\times10^{-5}$ | $1.84458477\times10^{-7}$ | pass |
| Torch/JAX vs pinned LAL quotient after merger, equal mass | absolute $5\times10^{-5}$ | $0.000153757344$ | fail |
| Torch/JAX vs pinned LAL quotient after merger, unequal and exchanged | absolute $5\times10^{-5}$ | $0.000189164604$ | fail |

The native signed tidal amplitude has the factor $M_{\rm sec}^2/(2D_{\rm sec})$
times the same dimensionless $-9\kappa_{\rm eff}^T x^{13/4}P(x)$ in both
backends. Equation (24) instead multiplies that dimensionless term by
$\sqrt{5\pi\eta/24}\,M_{\rm sec}^2/D_{\rm sec}$. The measured scalar ratios
are exactly the ratios of these factors. Public waveform outputs show that
Torch/JAX and the pinned LAL binary agree very closely before merger (for
example, at 1200 Hz in the equal-mass case, both give BNS/BBH modulus
$0.918650$ while literal Eq. (24) gives $0.934187$). Thus the **Eq. (24)
absolute correction and the carrier quotient have different normalizations
unless another convention explains the relation**. The public binary alone
does not establish why. This diagnostic does not declare either convention
correct or promote the waveform family.

At 100, 300, 800 and 1200 Hz, public LAL and native *absolute* BBH
amplitudes agree within $8.6\times10^{-15}$ relative error in the equal-
and unequal-mass systems. Their untapered BNS amplitudes agree within
$4.6\times10^{-8}$ relative error at those frequencies. A hidden overall
normalization of the LAL BBH carrier therefore does not explain the
Eq. (24) discrepancy.

The scalar merger frequency and Planck window pass for all four systems,
as do the native tapered/untapered window ratios. Zero tide passes every
binary check, including taper endpoints. The nonzero-tide native/LAL
difference within the taper remains unresolved; it is reported separately
from the larger Eq. (24) normalization discrepancy.

An additional read-only diagnostic isolates the shape of that small taper
difference. Divide the public LAL BNS/BBH modulus quotient by the native
*untapered* BNS/BBH modulus quotient. If the untapered corrections keep
agreeing inside the taper, the result is the LAL effective window. Invert
the published Planck window at $f=1.1f_{\rm mrg}^{\rm paper}$ to obtain an
equivalent merger boundary:

| System | Paper/native $f_{\rm mrg}$ (Hz) | Equivalent boundary (Hz) | Relative difference |
|---|---:|---:|---:|
| Equal | 2037.930727 | 2037.878683 | $-2.554\times10^{-5}$ |
| Unequal / exchanged | 1709.953767 | 1709.910782 | $-2.514\times10^{-5}$ |
| Zero tide | 4291.579158 | 4291.579158 | 0 |

Holding each inferred boundary fixed predicts the other sampled interior
window points with maximum residual $3.11\times10^{-7}$ (equal) and
$1.77\times10^{-7}$ (unequal). This strongly localizes the *public-output*
difference to an effective taper-boundary shift, but does not establish
whether a merger-frequency convention, an untapered amplitude difference
inside the window, or another binary detail caused it. No correction was
fitted to production code; both taper-specific frozen acceptance limits
remain failed in that diagnostic.

Execution used host `len`, cwd
`/home/xangma/tmp/diffgw-nrtidalv2-phase-20260926-e4a2`, CPU FP64,
LALSuite 7.26.15 (`lal` 7.7.1, `lalsimulation` 6.2.1), Torch 2.14.0+cpu,
JAX 0.11.2 and NumPy 2.5.3. The first, unmarked frozen gate returned
**21 failed, 19 passed**. After marking only the known nonzero-tide
failures `xfail(strict=True)`, moving native amplitude failures to a diagnostic
file outside the required `test_*math_reference*.py` CI job, and splitting out
independently passing merger/window and pre-taper binary checks, this exact
command returned
**36 passed, 24 xfailed** in 16.11 s:

```sh
PYTHONPATH=/home/xangma/tmp/diffgw-nrtidalv2-phase-20260926-e4a2/src:/home/xangma/tmp/diffgw-nrtidalv2-phase-20260926-e4a2/tests:/home/xangma/diffgw-pr45-318b37f7/.oracle-deps \
DIFFGW_REQUIRE_LAL_BINARY=1 OMP_NUM_THREADS=1 \
/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q -s \
tests/test_nrtidalv2_amplitude_math_reference.py \
tests/test_nrtidalv2_amplitude_diagnostic.py \
tests/test_nrtidalv2_amplitude_lal_binary.py
```

The logs are `amplitude_qualification.log` and
`amplitude_qualification_xfail.log` in that cwd. An earlier background
diagnostic PID 214583 has exited; the final command ran in the foreground.
Re-running a strict expected-failure case that
starts passing yields an XPASS failure, requiring the marker and this
receipt to be reviewed. This is a diagnostic gate, not a qualification.
