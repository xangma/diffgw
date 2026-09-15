# Pass 16: IMRPhenomXHM phase-node selection and phase gauge

Date: 2026-09-13. This is a bounded clean-room audit of the primary
IMRPhenomXHM source. The source facts were derived before inspecting the
current runtime. I did not inspect translated implementations, generated C,
notebooks, git history, verification output, residuals, or the
2026-09-11 specification directory. No runtime file or test was changed.

## Decision

For the non-mixing modes `(2,1)`, `(3,3)`, and `(4,4)`, the current runtime's
default node choice `[1,2,3,4,6]` is source-consistent. The paper's Fig. 8
calls the six available points `[C1,C2,E2,E3,E4,E5]` and identifies the
default five-point subset as `[C1,C2,E2,E3,E5]`; Table VII's `f1` through `f6`
therefore map this to:

```text
default nodes:       [f1, f2, f3, f4, f6] = [1, 2, 3, 4, 6]
default targets:     [p1, p2, p3, p4, p6]
```

The paper also specifies a parameter-dependent sparse-calibration subset:

```text
sparse subset:       [C1,C2,E3,E4,E5] = [f1, f2, f4, f5, f6]
                      = [1, 2, 4, 5, 6]
targets:              [p1, p2, p4, p5, p6]
```

It says to use this subset where the calibration has fewer waveforms,
described qualitatively as high-spin and/or low-`eta` regions. The primary
paper supplies no parameter inequalities, mask, table, or ancillary metadata
that identifies that region. Consequently, the current unconditional
`[1,2,3,4,6]` implementation is correct as the published default but does
not implement the paper's complete parameter-dependent selection. A new
switch cannot be reconstructed exactly from the primary source alone.

The `(3,2)` phase is a different solve. Its six-coefficient ansatz keeps
`a_3` and uses four collocation targets `p1` through `p4`, plus the value and
first-derivative conditions at `f_RD`; it is not a five-node `[1,2,3,4,6]`
solve. The current runtime's separate `(3,2)` branch uses `c_pts[:4]`, which
matches this source description.

## Primary derivation

Section V.B, Eqs. (5.4)–(5.5), gives the intermediate phase-derivative
ansatz

\[
 \frac{d\phi^{\rm Int}_{\ell m}}{df} =
 a_\lambda^{\ell m}\frac{f_{\rm damp}^{\ell m}}
 {(f_{\rm damp}^{\ell m})^2+(f-f_{\rm ring}^{\ell m})^2}
 +\sum_{k=0}^{4}\frac{a_k^{\ell m}}{f^k}.
\]

For `(2,1)`, `(3,3)`, and `(4,4)`, the source sets `a3=0`, leaving five
unknown coefficients. Equation (5.5) says the five right-hand sides are the
phase derivative evaluated at the selected intermediate collocation
frequencies. Thus a missing point means that its collocation equation is
omitted and another available point is used; it does not mean that the
missing target is set to zero, that its fit is deleted, or that the phase is
renormalized.

Table VII gives the six frequencies (the notation is reproduced here to make
the index mapping explicit):

```text
f1 = beta(eta) fIns
f2 = ((sqrt(3)+2) beta(eta) fIns - (sqrt(3)-2) fEnd) / 4
f3 = (fEnd + 3 beta(eta) fIns) / 4
f4 = (fEnd + beta(eta) fIns) / 2
f5 = (3 fEnd + beta(eta) fIns) / 4
f6 = (7 fEnd + beta(eta) fIns) / 8
```

Here `beta(eta)=1+0.001(0.25/eta-1)`. For non-mixing modes,
`fEnd=f_ring`; for `(3,2)`, `fEnd=f_ring^22-0.5 f_damp^22`. The paper says
that one more point than strictly needed is retained for flexibility, and
that in poorly constrained regions one point near inspiral is replaced by a
flatter near-ringdown point. Fig. 8 makes the replacement precise as
`[f1,f2,f3,f4,f6] -> [f1,f2,f4,f5,f6]`; it does not state a numerical trigger.

For `(3,2)`, Eq. (5.4) retains `a3`, so there are six unknowns. The source
uses `i in [1,4]` for the four collocation equations and adds
\[
 d\phi^{\rm Int}_{32}/df(f_{\rm RD}^{32})
   =d\phi^{\rm RD}_{32}/df(f_{\rm RD}^{32}),
 \qquad
 d^2\phi^{\rm Int}_{32}/df^2(f_{\rm RD}^{32})
   =d^2\phi^{\rm RD}_{32}/df^2(f_{\rm RD}^{32}).
\]
The footnote explains that this is the only mode where the ringdown phase
retains physical relative-time information because mode mixing is present.

## Absolute and relative phase conditions

The source distinguishes the calibrated phase derivative from the absolute
phase gauge. Hybrid waveforms are aligned so that the peak of the `(2,2)`
mode of `psi4` is at `t=0` and `Phi_22(psi4,t=0)=0` (Sec. II). Odd-`m`
modes retain a possible phase shift by integer multiples of `pi`; the authors
deliberately do not resolve that ambiguity from noisy odd-mode numerical data.
They require a smooth inspiral-to-merger phase and use the post-Newtonian
relative phase prescription instead.

Equation (4.11) writes the inspiral phase as the rescaled quadrupole phase,
the PN relative-phase correction, a linear-in-frequency term, and a constant.
The linear coefficient is fixed by continuity with the intermediate phase.
After the derivative has been reconstructed, Eq. (4.13) fixes the remaining
constant by the low-frequency condition

\[
 \lim_{f\to0}\left(\Psi_{\ell m}-\frac{m}{2}\Psi_{22}\right)
 =\frac{3\pi}{4}\left(1-\frac{m}{2}\right).
\]

This condition is a relative phase gauge tied to the paper's Fourier and
tetrad conventions. It does not require each subdominant phase to vanish,
and it is separate from the `p_i` phase-derivative fit targets. The source
also states that the relative time shifts among modes are calibrated in the
intermediate region around merger. For non-mixing modes the ringdown phase is
constructed by rescaling the quadrupole phase; for `(3,2)` the mixed spherical
phase is calibrated directly.

The current runtime contains an inherited `delta_target` phase connection.
This audit establishes the source condition above, but does not certify the
runtime's conversion between the source's `Psi` convention and its internal
carrier `phi` convention. That conversion should be treated as a separate
phase-gauge review before enabling any experimental phase connection by
default.

## Current-runtime implications

The inspected non-mixing branch solves the five-equation system with
`[c_pts[0], c_pts[1], c_pts[2], c_pts[3], c_pts[5]]` and targets
`[phase_p1, phase_p2, phase_p3, phase_p4, phase_p6]`. This is exactly the
published default for all three non-mixing modes. The inspected `(3,2)` branch
uses the first four points and the two ringdown matching conditions, as
required by the source.

The remaining source-level gap is the sparse switch. A faithful future
implementation needs an author-supplied parameter mask or a defensible
reconstruction of the unpublished “fewer calibration waveforms” region. It
must select `[1,2,4,5,6]` and use `p5` when active; simply dropping `p3`,
setting it to zero, or applying amplitude-veto regions would not be supported
by this paper.

## Source and asset record

Primary paper: [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2),
revised 2020-11-03; published as
[Phys. Rev. D 102, 064002 (2020)](https://doi.org/10.1103/PhysRevD.102.064002).
The [arXiv TeX source](https://arxiv.org/src/2001.10914v2) was downloaded as
the v2 source archive (SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`). The
main source file `hmc2.tex` has SHA-256
`5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e`.

The arXiv ancillary index is at
[`anc/ParspaceFits/README`](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/README),
SHA-256 `368184fb645eb93dc04963902c61bd37fd6be8be69be1842abed7994c20684f4`.
It lists separate phase-fit files `p1` through `p6`; the p5 assets are the
ones needed by the sparse subset. The archived local p6 assets used for this
audit are direct copies of the author `.m` files:

| Mode | Local asset | Author URL | SHA-256 |
|---|---|---|---|
| 21 | [`3Dfit_p6_lm_21.m`](/Users/xangma/repos/torchwave/docs/reviews/2026-09-12-clean-room/source-assets/pass-14-phase/3Dfit_p6_lm_21.m) | [arxiv.org/src/.../3Dfit_p6_lm_21.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/21/3Dfit_p6_lm_21.m) | `5b3ab0e7f04092f99f8983b02f7cb985cf4bfd13b3e4022b16cf020d57987716` |
| 33 | [`3Dfit_p6_lm_33.m`](/Users/xangma/repos/torchwave/docs/reviews/2026-09-12-clean-room/source-assets/pass-14-phase/3Dfit_p6_lm_33.m) | [arxiv.org/src/.../3Dfit_p6_lm_33.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/33/3Dfit_p6_lm_33.m) | `7cb4273b6b46f4be4f82b5a48e03705fa0f1d832effd71c79ade845c2d91afe8` |
| 44 | [`3Dfit_p6_lm_44.m`](/Users/xangma/repos/torchwave/docs/reviews/2026-09-12-clean-room/source-assets/pass-14-phase/3Dfit_p6_lm_44.m) | [arxiv.org/src/.../3Dfit_p6_lm_44.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/44/3Dfit_p6_lm_44.m) | `6dd15b7cf9fded8c3cdd329c7535d688f3d25633a9508c00ebf7fad2b29d5c8f` |

The p6 `.m` files are author fit assets, not a specification of the sparse
switch. Their presence, together with the README's p1–p6 inventory, supports
the existence of both node-target families but does not supply the missing
selection mask.
