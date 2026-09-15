# Pass 16: IMRPhenomXHM phase gauge and nonmixing inspiral connection

Date: 2026-09-13. This is a primary-source audit of the phase convention and
the disabled `experimental_phase_connections` branch. I derived the source
equations before inspecting the native implementation. No runtime or test
file was changed. I did not inspect translated implementations, generated C,
notebooks, git history, verification output, residuals, public binaries, or
the 2026-09-11 specification directory.

## Decision

The source fixes the low-frequency relative phase by

\[
 \Delta\Psi_{\ell m} := \Psi_{\ell m}-\frac m2\Psi_{22}
 = \frac{3\pi}{4}\left(1-\frac m2\right)
 = -\frac{3\pi}{8}(m-2).\tag{1}
\]

The native positive-frequency mode is constructed in the same form as the
paper's positive-frequency \(\tilde h_{\ell,-m}=A_{\ell m}e^{-i\Phi_{\ell m}}\):
the projection coefficients implement the paper's Eq. (3), and the native
mode is `A * exp(-1j * psi)`. Therefore the m-dependent term in the current

```text
delta_target = (l % 2) * pi - (3*pi/8) * (m - 2)
```

is exactly the source value in (1), but only after omitting the additional
`(l % 2)*pi`. Equation (4.13) contains no ell-dependent term. The extra term
is a sign flip of every odd-ell positive-frequency mode. It is invisible to
the appendix-B diagnostic after doubling the phase, and for odd `m` it can be
absorbed by the paper's acknowledged arbitrary π ambiguity, but it is not
derived by Eq. (4.13). For `(3,2)` it is an even-m sign choice and is not
covered by that ambiguity. The current `(3,2)` `delta_target_32 = pi` is
therefore also not source-certified by Eq. (4.13).

The experimental path should remain disabled. Its C1 algebra is the right
formal construction, but the assembled branch is not source-complete:

* Eq. (4.12) requires \(\Lambda^{PN}\) for `q < 100` and the fitted
  \(\Lambda^{fit}\) for `q >= 100`. The branch always calls
  `_pn_phase_lambda`; it never dispatches to the available native
  `XHMParameterFits.phase_lambda_{21,32,33,44}` fits.
* The paper defines the PN correction as the phase of a generally complex PN
  amplitude, including its constant argument. The native helper retains only
  a frequency-linear term and its only explicit constant sign adjustment is
  the ad hoc `(2,1)` test at `f=0.008`. The paper supplies no such frequency
  threshold or conditional `a21_pn >= 0` rule, and no corresponding complete
  sign prescription for the other modes.
* Consequently, although
  `dphi_ins = dphi_int - carrier_dp - lambda_dp` gives value and derivative
  continuity for the particular helper being used, it does not establish the
  paper's absolute/relative gauge. Enabling it would silently import an
  unverified time gauge and, at high q, the wrong \(\Lambda\) branch.

The implementation decision is therefore: retain the disabled flag, and do
not change `imrphenom_xhm.py` in this audit. A future enablement needs an
explicit convention decision for the ell-parity signs and an exact q-based
`Lambda` dispatch, followed by source-derived tests for the phase constant
and C1 join.

## Primary derivation

The paper defines the Fourier transform as

\[
 \tilde h(f)=\int h(t)e^{-i2\pi f t}\,dt,
\]

and equatorial symmetry as

\[
 \tilde h_{\ell m}(f)=(-1)^\ell\tilde h_{\ell,-m}^{*}(-f).
\]

For positive frequency it models the mode supported there,

\[
 \tilde h_{\ell,-m}(f)=A_{\ell m}(f)e^{-i\Phi_{\ell m}(f)},
 \qquad A_{\ell m}(f)\geq0.\tag{2}
\]

The native `_swsh_projection_coefficients` uses

\[
 d_+=\tfrac12\left(Y_{\ell,-m}+(-1)^\ell Y_{\ell m}^{*}\right),
 \qquad
 d_\times=\tfrac i2\left(Y_{\ell,-m}-(-1)^\ell Y_{\ell m}^{*}\right),
\]

and multiplies these coefficients by `h_mode`. Thus the native `h_mode` is
the positive-frequency quantity in (2), even though `return_unprojected_modes`
labels that tensor with the positive `m` key. The native negative-m entry
then applies `(-1)**l * conj(h_mode)`, which is the same parity operation as
the source relation. This establishes the phase-sign mapping to the internal
`exp(-i * psi)` convention without importing an external implementation.

For the inspiral, the source writes

\[
 \tilde h_{\ell m}(f)=A_{\ell m}^{\rm SPA}e^{i\Psi_{\ell m}},\qquad
 \Psi_{\ell m}=2\pi f t_c-\phi_{\ell m}-\pi/4+\psi_0,
\]

with \(\psi_0=\pi\) in the adopted tetrad. It then gives

\[
 \phi_{\ell m}(f)=\frac m2\phi_{22}^{X}(2f/m)
 +\Lambda_{\ell m}(f)+d\phi^{\rm Ins}_{\ell m}f
 +\phi^{\rm Ins}_{\ell m}.\tag{3}
\]

The derivative coefficient is chosen by continuity with the intermediate
phase, and the remaining constant is fixed by (1). The source's only explicit
parity statement for the tetrad check is

\[
 \operatorname{mod}(2\Phi_{\ell m}-m\Phi_{22},2\pi)
 =\begin{cases}\pi,&m\text{ odd},\\0,&m\text{ even},\end{cases}
\]

which depends on `m`, not `ell`. It cannot be used to justify adding `pi` for
odd `ell` to the positive-frequency mode.

The source separately notes that odd-m numerical phases have an unresolved
multiple-of-π ambiguity and chooses a smooth PN prescription rather than
using those noisy numerical phases. It also requires the positive-frequency
amplitudes in (2) to be non-negative. These statements permit an explicit
odd-m sign convention, but they do not define the native `(l % 2)` rule or
the `f=0.008` `(2,1)` test.

## Experimental branch audit

The source's formal C1 construction is straightforward. If `P(f)` is the
rescaled quadrupole phase, `L(f)` is the selected \(\Lambda_{\ell m}\), and
`I(f)` is the intermediate phase, then at `f_ins`:

\[
 d\phi^{\rm Ins}=I'(f_{\rm Ins})-P'(f_{\rm Ins})-L'(f_{\rm Ins}),
\]

and the value shift into `I` is the difference between the resulting
inspiral value and `I(f_ins)`. The native branch implements this algebra for
the helper it calls. That fact proves C1 continuity of the assembled pieces,
not correctness of the source gauge.

Equation (4.12) is explicit: use \(\Lambda^{PN}\) when `q < 100` and
\(\Lambda^{fit}\) when `q >= 100`. The native `xhm_fits.py` contains
`phase_lambda_21`, `phase_lambda_32`, `phase_lambda_33`, and
`phase_lambda_44`, and those expressions agree with the author `Lambda.m`
ancillary assets. They are not called by the experimental branch, which
currently evaluates `_pn_phase_lambda` for every q. The existence of the
expressions makes the missing dispatch identifiable; it does not certify how
their fitted correction should be combined with a retained frequency-linear
term unless the source asset's representation is wired and tested explicitly.

The native helper also drops the constant PN argument. That constant matters
because (2) uses non-negative amplitudes: a negative real PN prefactor is a π
phase choice. The paper's source equations state the complex-amplitude
argument prescription but do not state the runtime's `a21_pn` probe frequency,
inequality, or branch for the other modes. This is a concrete source-level
blocker for enabling the experimental path, independent of the unresolved
ell-parity gauge.

## Sources and hashes

Primary paper: [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2),
revised 2020-11-03; published as
[Phys. Rev. D 102, 064002](https://doi.org/10.1103/PhysRevD.102.064002).
The official [arXiv TeX source](https://arxiv.org/src/2001.10914v2) was
downloaded from the v2 source archive, SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`;
`hmc2.tex` SHA-256
`5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e`.
The relevant ancillary fits are the official files
`anc/ParspaceFits/PhaseFits/{21,32,33,44}/Lambda.m`; their SHA-256 hashes are,
respectively,

```text
21  d8868c0d038bf69b3c9d3852930b1a8d635a85ff10cce25e7ad9999a280c4906
32  da5e09900972cb53b8ea3e419533a3e9058a8ab80b619b6f3a001b32bfcf6716
33  5e7da0c943cab189cb6a3d49a6db81e8b561ac98654c30e67e84baf334fd5461
44  c701d46ab1c7825582ef870747aec8b55c8113ee05f8416f5b0935891767be2d
```

Native files inspected after the primary derivation were
`src/torchwave/approximants/imrphenom_xhm.py` and
`src/torchwave/approximants/imrphenom_hm.py`; no changes were made to either.
