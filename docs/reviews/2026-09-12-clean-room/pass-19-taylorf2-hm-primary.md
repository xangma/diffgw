# Pass 19: TaylorF2_HM primary-source audit

Date: 2026-09-13

This audit covers the native `TaylorF2_HM` construction after deriving the
published post-Newtonian mode and stationary-phase formulas.  The source set
is limited to the original PN papers and the official supplementary
mathematical asset listed below.  I did not inspect LAL/PyCBC or other
implementations, translated/generated code, notebooks, git history, the old
clean-room specification, verification artifacts, public binaries, or
residuals.

## Source ledger

The primary papers were downloaded from their arXiv records.  The recorded
PDF SHA-256 values are:

| source | URL/version | SHA-256 |
| --- | --- | --- |
| Kidder, *Using Full Information When Computing Modes...* | [arXiv:0710.0614](https://arxiv.org/abs/0710.0614) | `539d5d9fee65efbab24779e16b511ff3300d33fec63528a1583ae54164df2bbf` |
| Arun, Buonanno, Faye & Ochsner, *Higher-order spin effects...* | [arXiv:0810.5336](https://arxiv.org/abs/0810.5336) | `9f55c91ecf974862798a81832540a602652803b9f14ca957bda711e63c818416` |
| Blanchet, *Post-Newtonian theory for gravitational waves* | [arXiv:1310.1528](https://arxiv.org/abs/1310.1528) | `c3b1b7c454e15b65bade4245a8e6d61b6e5c0546baaee436a17b99ef6983b4a8` |
| Mishra, Kela & Arun, *Ready-to-use post-Newtonian gravitational waveforms...* | [arXiv:1601.05588](https://arxiv.org/abs/1601.05588) | `e5287b50f0e44ebfc5156684217b00f73a5ab872f8f6a0e19884c24d3478840` |
| Henry, *Complete gravitational-waveform amplitude modes...* | [arXiv:2210.15602](https://arxiv.org/abs/2210.15602), v2 | `6a54d6f36cb103e0daaedbba31ba3ecd8cc527f8a9b963533156f0574a29ae24` |

The official Henry source archive was obtained from the arXiv e-print record
([export.arxiv.org/e-print/2210.15602](https://export.arxiv.org/e-print/2210.15602));
the archive SHA-256 is
`d656d47398cd46e2accec123b45ac3fd4ef22f4618f6aeb937912b415ac24d3d`.  The
mathematical asset `anc/modes_PNexp_full_35PN.dat.m` has SHA-256
`08bb11a4e551cd264235ba5fe573466e4b9e6fffb104ed0d68a565d0bed66137`.
Its header identifies it as the spin-weighted spherical-harmonic mode
expansion, including non-spinning and spinning contributions through 3.5PN
for quasi-circular aligned-spin binaries.  It is an equation/data asset, not
a waveform implementation.

## Published content and normalization

Arun et al. Eq. (6.7) writes the time-domain polarizations as a sum over
harmonics (k=1,ldots,7) and relative PN powers (n=0,ldots,7),

\[
h(t)=\frac{2M\nu}{D_L}\sum_{n=0}^{7}\sum_{k=1}^{7}V^{2+n}
 [\alpha_k^{(n)}\cos(k\Psi)+\beta_k^{(n)}\sin(k\Psi)]+O(V^8).
\]

Equation (6.8) defines (f_k=kF) and

\[
V_k=[2\pi M(f_k/k)]^{1/3}=(2\pi MF)^{1/3}.
\]

The SPA amplitude and phase are given in Eqs. (6.9)--(6.12), and the
frequency-domain sum in Eq. (6.13) is

\[
\tilde h(f)=\frac{M^2}{D_L}\sqrt{\frac{5\pi\nu}{48}}
 \sum_{n=0}^{7}\sum_{k=1}^{7}V_k^{n-7/2}C_k^{(n)}
 \exp\{i[k\Psi^{\rm SPA}(f/k)-\pi/4]\}.
\]

Equation (6.14) folds the SPA curvature factors into (C_k^{(n)}), including
the (S_2,ldots,S_5) terms from Eq. (6.10).  Thus the published mapping is
mode frequency (F=f/k), followed by multiplication of the orbital phase by
(k), with a common (-\pi/4) stationary-phase constant.  It is not a
single quadrupole phase evaluated at detector frequency without this mapping.

There is an additional normalization uncertainty at this interface. Arun's
Eq. (6.8) uses \(V_k=(2\pi M f/k)^{1/3}\) for each harmonic, while the native
code computes one shared v22=(pi*M*f)^(1/3) and one quadrupole amp0 (lines
208--227), then builds every mode from those quantities and a ratio (lines
258--275). Some explicit leading ratios contain powers of 2 or \(m\), but the
allowed sources do not establish that the four native constants absorb all
harmonic \(V_k\) powers, the SPA Jacobian, and the corresponding frequency
argument relative to the shared quadrupole amplitude. Each ratio would have
to be re-derived from Eqs. (6.8), (6.10), (6.13), and (6.14) before adding a
mode or correcting one coefficient. This is a separate reason not to apply a
piecemeal amplitude patch.

Kidder gives explicit time-domain mode expressions, including (h^{21}),
(h^{22}), (h^{33}), and (h^{44}), in the mode equations in the body of
the paper.  Arun's Appendix D gives the corresponding explicit harmonic
coefficients.  Henry's official asset is broader: expression blocks begin at
asset lines 28, 31, 113, 196, 278, 340, and 432 for (h[2,0]), (h[2,1]),
(h[2,2]), (h[3,1]), (h[3,2]), (h[3,3]), and (h[4,0]), and continue
through ℓ=8.  In particular, (h[3,2]) is explicitly published.  The
four native modes therefore form a supported subset, not the complete
published harmonic content.

The leading mode normalization used by the native code,
`sqrt(16*pi/5) * A0`, is compatible with expressing the leading (22) mode
through the code's spin-weighted spherical-harmonic projection.  The native
ratios in `taylorf2_hm.py:258-273` are leading-order ratios.  The source
amplitudes contain more terms: Kidder supplies non-spinning mode corrections
through the stated mode-dependent orders (with (h^{22}) extending to 3PN),
Arun supplies non-spinning Fourier amplitude terms through 2.5PN plus
spin-orbit terms through 1.5PN and spin-spin terms through 2PN, and Henry's
asset supplies complete non-precessing mode expressions through 3.5PN.  No
single one of those truncations is selected by the native API, so replacing
the four ratios with “the” published correction would require an unstated
truncation choice and all associated complex coefficient terms.

## Observer, symmetry, and spin conventions

Kidder, Arun, and Mishra use the complex strain convention

\[
h=h_+-ih_\times=\sum_{\ell,m}h^{\ell m}\,{}_{-2}Y_{\ell m},
\qquad h^{\ell,-m}=(-1)^\ell\overline{h^{\ell m}}
\]

for non-precessing systems, with the observer angles carried by the
spin-weighted harmonics.  The native projection at lines 20--88 implements
this negative-(m) completion for its four positive-(m) modes.  The source
mode formulas also show why odd-(m) behavior cannot be represented by the
mass difference alone: the leading non-spinning (21) term is proportional
to Δ, while the Henry `h[2,1]` block at lines 31--112 also contains a
spin-difference Σ contribution.  Consequently, equal masses with unequal
aligned spins can have an odd-(m) mode.  The native `21` and `33` amplitudes
use only `delta` (lines 262--269), so they omit these published spin
amplitude terms.  Repairing that omission requires choosing the Arun or Henry
amplitude truncation and translating every relevant Σ/spin term into the
native normalization and phase convention; one isolated coefficient would
not be source-complete.

The native phase uses `exp(-i*psi_lm)` and identifies the result as
(h_{\ell,-m}) (lines 292--294).  Its offset

\[
\Delta\Phi_{\ell m}=\frac{3\pi}{4}(1-m/2)
\]

is a model-specific convention cited in the native comment (line 254).  The
allowed original PN source gives the Eq. (6.13) common (-\pi/4) SPA term,
but does not identify this particular offset for the native (h_{\ell,-m})
storage convention.  Algebraically, applying Eq. (6.13) to a base phase that
already contains (-\pi/4) would produce an (m)-dependent conversion

\[
\psi_m=\frac m2\psi_{22}(2f/m)+\left(\frac m2-1\right)\frac\pi4,
\]

whereas the native offset is different.  The source set does not establish
that these two stored-mode conventions are identical, so changing the phase
offset would be a convention guess and is not made in this pass.

## Spin, tides, and cutoff

Arun's ready-to-use waveform assumes aligned or anti-aligned spins and states
the amplitude spin content at the orders above.  Henry extends the mode
amplitudes to 3.5PN for aligned spins.  These papers do not provide tidal
corrections to the higher-mode amplitude blocks audited here.  The native
implementation forwards `lambda1` and `lambda2` into the underlying TaylorF2
phase (lines 239--250), while its higher-mode ratios remain point-particle
leading terms.  That is a documented capability boundary, not evidence for
tidal higher-mode amplitude terms.

The source relation (f_k=kF) implies a harmonic cutoff obtained from an
orbital cutoff as (f_{{\rm cut},k}=kF_{\rm cut}).  The papers do not prescribe
the native optional Schwarzschild/Kerr ISCO truncation.  When enabled, the
native rule `f_cut_m=(m/2)*f_ISCO_22` (lines 191--207 and 277--290) is exactly
the algebraic (mF_{\rm ISCO}) conversion if `f_ISCO_22=2F_ISCO`; its choice
to use ISCO, and the smooth roll-off policy, are API policy rather than a
published TaylorF2_HM equation.  No cutoff change is justified from these
sources.

## Native comparison, decision, and validation

The native model supports only `(2,2)`, `(2,1)`, `(3,3)`, and `(4,4)`
(`taylorf2_hm.py:91-106`), uses Newtonian amplitude ratios
(`:258-275`), passes spins and tides to the base phase (`:239-250`), and
applies the harmonic ISCO policy described above.  The audit therefore finds
source-complete omissions: published higher modes such as `(3,2)`, PN
amplitude corrections, and spin-difference odd-mode terms are absent.

No runtime or test change is made.  A correct implementation requires a
declared amplitude truncation (Arun 2.5PN versus Henry 3.5PN), the complete
coefficient set at that truncation, and an explicit reconciliation of the
published SPA phase convention with the native negative-(m) storage and
projection.  The allowed sources do not determine that API choice.  Adding a
mode or changing a phase constant without it would invent metadata and could
silently change observer/sign conventions.  The source-complete result for
this pass is therefore **documented blocker; no patch**.

No ordinary model tests were run because no runtime/test files changed.  The
review file was checked for whitespace errors with `git diff --check` (and an
untracked-file check against `/dev/null`).
