# TaylorF2_HM leading eight-mode mathematical specification

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `TaylorF2_HM` |
| Reference composition | `taylorf2-hm-leading-carrier/v1` |
| Scope | Eight leading nonspinning SPA mode amplitudes with inherited aligned-spin TaylorF2 phase and negative-m polarization projection. |
| Domain and modes | `frequency`; `(2, 2)`, `(2, 1)`, `(3, 1)`, `(3, 2)`, `(3, 3)`, `(4, 2)`, `(4, 3)`, `(4, 4)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 0 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `spin_order` | `7` | `-1`, `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8` | None |
| `f_isco_cutoff` | `true` | `false`, `true` | None |
| `smooth_cutoff` | `false` | `false`, `true` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

Specification: `taylorf2-hm-leading-carrier/v1`. This scalar SymPy/mpmath
reference covers the native frequency-domain carrier's eight modes
$(2,2),(2,1),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4)$. It composes leading
nonspinning mode amplitudes, the [separately versioned aligned-spin TaylorF2
phase](taylorf2-aligned-spin.md), and positive-frequency polarizations. The
mode amplitudes are **not** full-PN or spin-corrected mode expansions. The
mathematical domain is positive frequencies and positive component masses;
slow inspiral is the physical regime in which the approximation is motivated.

## Inputs, conventions, and options

The public masses and distance are in nominal solar masses and Mpc. At the
reference boundary, $M=(m_1+m_2)T_\odot$ and $D=D_{\rm Mpc}\,\mathrm{Mpc}/c$
are seconds, $\delta=(m_1-m_2)/(m_1+m_2)$, and
$\eta=(1-\delta^2)/4$. $f$ and optional $f_{\rm ref}$ are Hz; $t_c$ is seconds;
$\iota$, $\phi_c$, and observer azimuth $\phi_0$ are radians. Spins $\chi_1$,
$\chi_2$ are aligned dimensionless projections. The scalar kernel has no
production imports or unit constants.

The stored positive-frequency mode is $h_{\ell,-m}(f)$, and the Fourier
carrier is $e^{-i\Psi_{\ell m}}$. Set
$Y_{\ell m}={}_{-2}Y_{\ell m}(\iota,\phi_0)$ with
$Y_{22}=\sqrt{5/(64\pi)}(1+\cos\iota)^2e^{2i\phi_0}$.
The scalar reference constructs these harmonics with Wigner $d$ matrices,
independently of the frontends' trigonometric polynomials. With the native
positive-frequency convention,

$$
d_+={Y_{\ell,-m}+(-1)^\ell Y_{\ell m}^*\over2},\qquad
d_\times={i\over2}\bigl(Y_{\ell,-m}-(-1)^\ell Y_{\ell m}^*\bigr),\qquad
(h_+,h_\times)_{\ell m}=(d_+,d_\times)h_{\ell,-m}.
$$

This identification is not inferred from frontend agreement.
[García-Quirós et al., Eqs. (2), (4)–(9)](https://arxiv.org/html/2001.10914)
use $h=h_+-ih_\times$ and the LAL Fourier sign
$\tilde h(f)=\int h(t)e^{-i2\pi ft}dt$, show that $m>0$ modes are
concentrated at negative frequency, and define the positive-frequency object
$\tilde h_{\ell,-m}=A_{\ell m}e^{-i\Phi_{\ell m}}$ before deriving the two
projection factors above. The public mode dictionary retains positive
$(\ell,m)$ labels for these paired polarization contributions; it does not
expose raw positive-$m$ Fourier modes.

The default native `spin_order=7` selects `SO3, SS4, SO5, SO6, SS6, SO7` in
the [aligned-spin option map](taylorf2-aligned-spin.md#frontend-option-contract).
The scalar formulas can also select other named term sets, including the full
set used by native `spin_order=8`. The direct complex comparison uses
`f_isco_cutoff=False`, `lambda1=lambda2=None`; only the Schwarzschild hard
cutoff threshold is additionally checked. Native Kerr ISCO, half-cosine taper,
and tidal terms are outside this reference slice. Native hard-cutoff masks are
piecewise and have no derivative claim at the boundary.

## Mathematics and components

Let $A_0(f)$ be the inherited restricted TaylorF2 face-on amplitude,
$v=(\pi Mf)^{1/3}$, and $I(v)$ its aligned-spin intrinsic phase with the
declared term set. Define $q_m=2/m$ and

$$
\Psi_{22}^{(m)}=I\bigl((\pi Mq_m f)^{1/3}\bigr)
 +2\pi q_m f t_c-2\phi_c-\pi/4,
\quad
\Psi_{\ell m}={m\over2}\Psi_{22}^{(m)}
 +{3\pi\over4}\left(1-{m\over2}\right),
\quad
h_{\ell,-m}=\sqrt{16\pi/5}\,A_0(f)R_{\ell m}(v,\delta)
 e^{-i\Psi_{\ell m}}.
$$

The phase adapter follows [García-Quirós et al., Eqs. (29),
(34)](https://arxiv.org/html/2001.10914): the low-frequency SPA relation is
$\Phi_{\ell m}\simeq(m/2)\Phi_{22}(2f/m)$ and its convention-dependent
constant is $3\pi(1-m/2)/4$. Consequently $t_c$ remains a common
$e^{-i2\pi f t_c}$ shift while $\phi_c\mapsto\phi_c+\Delta\phi_c$ multiplies
the native carrier by $e^{im\Delta\phi_c}$. Eqs. (30)–(32) of that paper add
a frequency-dependent phase for complex higher-PN amplitudes; omitting it is
justified only for this declared leading real-amplitude slice.

When `f_ref` is supplied, subtract $I((\pi Mq_m f_{\rm ref})^{1/3})$
inside $\Psi_{22}^{(m)}$ and replace $-\pi/4$ by $+3\pi/4$.
This is the inherited project phase origin and scales the reference frequency
by $q_m$. The normalized real leading-amplitude ratios are

| Mode | $R_{\ell m}$ |
| --- | --- |
| 22 | $1$ |
| 21 | $\delta v/(3\,2^{1/3})$ |
| 31 | $\delta v/(12\sqrt7\,2^{5/6})$ |
| 32 | $\sqrt{5/7}(1-3\eta)v^2/3$ |
| 33 | $(3/4)\sqrt{5/7}(3/2)^{5/6}\delta v$ |
| 42 | $\sqrt5(1-3\eta)v^2/63$ |
| 43 | $(3/4)\sqrt{3/35}(3/2)^{1/6}\delta(1-2\eta)v^3$ |
| 44 | $(8/9)\sqrt{5/7}(1-3\eta)v^2$ |

These ratios use the *observed-mode* $v$. In the independent kernel, the
printed $\mathcal H_{\ell m}(x)$ of Eq. (21) is evaluated at
$x_m=(2/m)^{2/3}v^2$ and multiplied by $(m/2)^{2/3}$ after dividing Eq. (30)
by its $(2,2)$ Newtonian amplitude. A fixed unit-complex factor for each mode
converts the source's complex coefficient to the declared real-amplitude
carrier convention. [Mezzasoma and Yunes, Eqs. (20)–(21i), (23)–(30)](https://arxiv.org/html/2203.15934)
provide an all-eight-mode leading-amplitude construction;
[Mishra et al., Eqs. (9)–(12h)](https://arxiv.org/html/1601.05588) give a
second all-eight-mode frequency-domain construction. Their arXiv v1 source
bundle also contains the author-published Mathematica supplement
`supl-mkaf16.m`, whose `H[l,m]` definitions cover every mode in this
specification through 2PN amplitude order. Both papers use the opposite
Fourier sign, so their printed complex modes cannot be compared directly with
the stored negative-$m$ mode without the Fourier, tetrad, and per-mode phase
adapter described above. García-Quirós et al. fix the native convention and
give seven of the mode amplitudes in Appendix E; the $(4,2)$ ratio uses the
first two all-eight sources. García-Quirós et al. also report a higher-PN
amplitude disagreement with the earlier calculation, beyond this leading-order
slice.

For the optional Schwarzschild hard mask, mode $m$ has
$f_{\rm cut,m}=m/(2\,6^{3/2}\pi M)$; the native carrier retains
$0<f\le f_{\rm cut,m}$. This is a project truncation, not a source amplitude.

The stable IDs are `tf2.hm.ratio_LM.v1` and
`tf2.hm.projection_LM.v1` for every listed mode, plus
`tf2.hm.mode_phase.v1`, `tf2.hm.polarizations.v1`, and
`tf2.hm.cutoff_frequency.v1`. All are executable in the stated carrier scope.
The ratio, projection, phase, and polarization comparisons cover both
frontends; the cutoff ID covers the threshold expression and hard-mask samples.

## Implementation mapping

The scalar functions `amplitude_ratio`, `spin_weighted_harmonic`, `projection`,
`mode_phase`, `mode_expr`, `waveform_expr`, and `cutoff_frequency_expr` live in
[`taylorf2_hm.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/taylorf2_hm.py). Native mode
amplitudes, phases, and projections live in the separate
[`torchwave`](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/taylorf2_hm.py) and
[`jaxwave`](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/taylorf2_hm.py) implementations.
`return_modes=True` exposes per-mode projected $(h_+,h_\times)$, which the
reference checks before testing their sum. The scalar reference imports
neither native implementation; the native implementations import no scalar
reference.

## Evidence and limitations

The evidence is intentionally separated:

| Evidence layer | Maintained check | Establishes | Does not establish |
| --- | --- | --- | --- |
| Author-supplement components | `test_taylorf2_hm_author_supplement_math_reference.py` | Checksum-pinned leading $|h_{\ell m}/h_{22}|$ plus signed, absolutely normalized nonspinning complex raw modes for all eight modes in Torch and JAX | External numerical parity, aligned-spin composition, `f_ref` subtraction, projection, cutoffs, or the complete waveform |
| Paper fidelity | `test_taylorf2_hm_paper_math_reference.py` | All-eight native $h_{\ell,-m}$ reconstruction, Eqs. (8)–(9), the $2f/m$ map, SPA constant, and $t_c/\phi_c$ signs | An independent executable waveform |
| Scalar-reference agreement | `test_taylorf2_hm_math_reference.py` | All-eight leading amplitudes, composed aligned-spin phase, complex polarizations, selected derivatives and hard cutoffs | External parity or physical accuracy |
| Torch/JAX parity | `tests/parity/test_hm_parity.py` | Cross-backend values, individual modes, JIT and autodiff at sampled points | Independence from shared equations |
| External angular/API boundary | `test_taylorf2_hm_external_api.py` | Public LAL spin-weighted harmonics for every supported $\pm m$; public TaylorF2 does not return resolved higher modes | Any higher-mode TaylorF2 waveform value |
| External waveform parity | frozen `TaylorF2_HM_diagnostic` | Dominant $(2,2)$ carrier only | The other seven modes |
| Physical validation | None | — | Accuracy against NR, EOB, or observations |

The scalar-reference test compares all eight complex per-mode polarizations and
their sum at two frequencies, two mass ratios, both zero and aligned spin, two
orientations, and both phase origins, in FP64 CPU Torch and JAX against an
80-digit scalar evaluator. It also checks each mode's azimuth derivative,
a component-mass derivative, equal-mass suppression of odd modes, and
Schwarzschild hard-mask thresholds. The per-mode normalized complex tolerance
is $4\times10^{-10}$ and the derivative tolerances are stated in the test.
These are sampled implementation checks, not a uniform error bound.

The author-supplement fixture is generated without native imports from the
lowest nonspinning power of each pinned `H[l,m]`. At common observed frequency,
its explicit adapter is
$V_m=(2/m)^{1/3}v$ and
$|h_{\ell m}/h_{22}|=|H_{\ell m}^{\rm LO}(V_m)|(V_m/v)^{-7/2}$.
The source bundle SHA256 is
`f0b6c6810da95f45141dbb9f0c5cccec0ba0742f60b836dba7698c90a25d5154`;
the supplement SHA256 is
`501ea53c29b77e061585540b9ccfd69a4cd0283e805ea1e9c24c38800e627a70`.
The project stores only the generated fixture and provenance, not the author
source asset. The fixture's normalized `TF28Phasing` is nevertheless a
transformed author-supplement expression, not an independently authored
numerical result; its downstream redistribution terms remain unresolved.

The same pinned supplement contains `TF28Phasing`. Its nonspinning part is an
orbital SPA phase with prefactor $3/(256\eta v^5)$; the factor of $m$ in
Mishra et al. Eq. (11) makes it identical to the native $(m/2)I(V_m)$ phase.
This also resolves the factor-of-two ambiguity introduced by the
$3/(128\eta v^5)$ prefactor printed in the paper's Eq. (4): the maintained
component gate follows the checksum-pinned author expression used in Eq. (11).
The generator retains the normalized SymPy expression, and the test combines
it with the signed leading $H_{\ell m}$ term and Eq. (11) normalization. The
fixed raw-mode convention adapter is
$h^{\rm native}_{\ell,-m}=e^{i\pi(m-3)/2}h^{\rm Mishra}_{\ell,m}$.
This gate is restricted to positive-frequency, leading nonspinning,
unwindowed raw modes with `f_ref=None`. Projection is used only to recover the
native raw carrier and is not attributed to the supplement.

Exact all-mode external complex parity remains unprovable from the available
public sources. The supplement is symbolic author input parsed by this
project, not an independently maintained executable or numerical series. It
supplies full 2PN amplitudes and a phase expression while the maintained gate
selects the leading nonspinning amplitudes; it does not emit the native
negative-$m$ series with aligned-spin composition, `f_ref` subtraction,
paired polarization projection, and per-mode cutoff. No author-published
numerical complex arrays fix that complete option set.
The bounded independent-oracle search and its exact acceptance fields are
recorded in the [2026-09-26 evidence receipt](../validation/taylorf2-hm-oracle-search-20260926.md).
Public LAL TaylorF2 returns one frequency series and its public FD-mode
dispatcher has no TaylorF2 generator. IMRPhenomXHM is a different full-IMR
model and `gwfast`'s TEOBResumSPA uses different EOB dynamics and amplitude
corrections, so neither is a binwise same-model oracle. Higher-PN mode
amplitudes, spin-amplitude corrections, generic tidal matter, merger, Kerr
cutoffs, taper derivatives, FP32, and full-domain accuracy remain unqualified.

The exact next external-parity gate is an author-supplied numerical table or
an independently maintained executable that exposes all eight complex
$h_{\ell,-m}(f)$ values for a fully specified binary and option set. Closure
would require direct, unaligned binwise comparison after applying only the
published Fourier/tetrad adapter above; a dominant-mode series or angular
basis alone cannot satisfy that gate.

## Reproduce the checks

Install `python -m pip install -e '.[reference,torch,jax,dev]'`, then run
`python -m pytest -q tests/test_taylorf2_hm_author_supplement_math_reference.py tests/test_taylorf2_hm_math_reference.py tests/test_taylorf2_hm_paper_math_reference.py`
for the mathematical slice. With pinned LALSuite 7.26.15 installed, run
`python -m pytest -q tests/test_taylorf2_hm_external_api.py` for the public
angular/API boundary.
To regenerate the frozen supplement fixture, download
`https://export.arxiv.org/e-print/1601.05588v1` and run
`python tests/data/generate_taylorf2_hm_mishra2016_samples.py --archive PATH`;
the generator rejects either source hash before parsing and imports no native
waveform code.
The complete CPU gate is `python -m pytest -q --require-math-reference`.
That gate requires both frontends and reference dependencies and rejects
skips, expected failures, and deselection. Run
`python scripts/generate_waveform_library_table.py --check` to verify the
catalog-derived sections. Equation, convention, or cutoff changes require a
new component/specification version and a reviewed source and test update.
