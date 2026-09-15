# SEOBNRv4 integration assessment

Date: 2026-09-12. Read-only code assessment; no implementation changes or
new numerical-validation claim. Primary equations and author publications
only; no external implementation, oracle output, or old calibration spec.

## Decision

**There is substantial actionable EOB implementation work, but the present
components cannot yet produce an unambiguous complete published SEOBNRv4
(2,2) waveform.** Full spinning flux amplitudes and the trajectory/NQC/
merger integration are missing. Their primary equations are available.
Separately, the inspected papers do not settle the complete inherited
`delta22` expression or which of Hofmann's three final-spin coefficient
blocks SEOBNRv4 selected. These are convention/selection questions, not
unavailable coefficient tables.

An explicitly labeled reconstruction that declares those choices could be
built. It would not establish the uniquely intended SEOBNRv4 prescription
or implementation parity. Conversely, the unresolved choices do **not**
make all remaining EOB development impossible. Any broader requested
parity claim remains contingent on every model's inputs and conventions;
completing this EOB work alone would not establish it.

## Current component boundary

| Component | Current state and required integration |
|---|---|
| Conservative dynamics | `SEOBNRv4Hamiltonian` supplies the published calibration polynomials and a primary-equation reconstruction. Its documented repairs to the radial potential and tortoise quartic expression remain explicit interpretations; see pass-3-seobnrv4.md. The waveform driver exposes `conservative_model='seobnrv4_published'`, but that switch alone does not select a v4 waveform. |
| Initial conditions | The current circular solve plus leading dissipative radial momentum implements the aligned specialization of Buonanno–Chen–Damour (2006), Sec. IV A, Eqs. (4.8)–(4.15). It can consume the eventual consistent flux. This is not an unavailable physical input. |
| Factorized modes/flux | DIN nonspinning modes cover all 35 positive-m modes through ell=8. The optional Pan leading-spin `f` correction stops at ell=4 and is not the inherited v4 spinning amplitude prescription. |
| New (2,2) amplitude | `seobnrv4_published_amplitude22` provides the reconstructed modulus, non-Keplerian Newtonian factor, source and tail modulus. It is not used by the waveform/flux driver and intentionally supplies no residual phase. Its interpretation of Bohé's replacement of the old v^7 spin coefficient remains documented. |
| NQC | Target fits, time offset, bases, and the two matching systems exist. The caller still must locate the orbital-frequency peak, supply a resolved local stencil and the **uncorrected complex-mode GW frequency**, and apply the solved correction after evolution. |
| Merger/ringdown | The Bohé ansatz and shape fits exist. The caller must supply the remnant QNM and matching data at the calibrated amplitude peak. |
| Driver | It still stops at the integrator's radius/time limits, attaches a simple exponential at the last retained sample, and uses the PhenomD remnant prescription. It does not call the new v4 amplitude, NQC, or merger modules. |

The current fixed-step integrator has no orbital-frequency-peak refinement
or per-system termination for a heterogeneous batch. A sample immediately
before a radius cutoff is not the calibrated matching event. These are
engineering gaps, not missing calibration data.

## Spinning flux is available work; phase is a separate question

Bohé Sec. II B inherits Taracchini (2014)'s factorized waveform prescription,
with its stated (2,2) amplitude replacements. The necessary primary chain is:

- Taracchini (2012), Appendix A: non-Keplerian prefactor, parity-dependent
  source, tail, amplitudes through ell=8, and regular odd-m treatment.
- Pan (2011), including its cited erratum: comparable-mass spin terms and
  the higher test-particle spin terms in `f_lm`/`rho_lm`.
- Buonanno–Faye–Hinderer (2013): comparable-mass spin contributions through
  the order adopted by Taracchini (2014).
- Taracchini (2014), discussion of Eq. (4): hybridization, the odd-m
  prescription, and removal of the old adjustable nonspin amplitude term.
- Bohé (2017), Eqs. (2.4)–(2.5): the stated (2,2) amplitude replacements.

Reconstructing, transcribing and checking this whole amplitude set remains
work. The existing DIN set plus a leading-spin additive correction does not
complete it. The reviewed chain does not reveal an unavailable fitted
amplitude table: this is primarily missing algebra/implementation, with
each hybridization/replacement convention to be recorded explicitly.

Crucially, Bohé Eq. (2.9) uses squared **moduli**. For positive-m modes in
the native M=1 normalization,

```
F_energy = sum_(ell=2..8,m=1..ell) (m*Omega)^2 |R*h_lm/M|^2 / (8*pi)
F_phi = -F_energy/(nu*Omega)
F_r_star = (p_r_star/p_phi)*F_phi.
```

No higher-mode residual phases are needed for this flux. NQC is explicitly
excluded from radiation reaction in v4. Thus unknown complex phases above
ell=4 do not block implementing spinning radiation reaction through ell=8.

For the emitted complex (2,2) mode, the existing pass-3 mode22 report's
specific questions remain unresolved by the fresh Bohé reading:

1. Does the inherited positive-chi term `540*nu*chi*v^6` remain?
2. Does the v2 import of test-particle spin effects include Pan Eq. (27a)'s
   residual phase? If so, how is its `-4*qK*v^6/3` term combined with
   Bohé Eq. (2.6), and is `20*qK*v^8/63` retained?
3. Eq. (2.6) labels its correction 3PN but uses `H_EOB`, defined with rest
   mass subtracted in Eq. (2.1). Total real energy restores the stated PN
   order and test-particle limit; this is a justified repair, not a literal
   equality of the two printed definitions.

Literal inheritance suggests retaining the 540 term, but does not uniquely
settle the entire phase. A consolidated author expression specifying its
variables and inclusion/replacement rules would close this boundary.
The complex tail must also use Taracchini Appendix A's
`r0=2M/sqrt(e)`; the existing DIN helper's `r0=2M` is a different phase
convention. The old helper also omits the v^9 term of Taracchini A16.
Both tail radius and that nonspin term are printed and implementable.

## Compatible remnant inputs

### Final mass: resolved primary formula, currently unimplemented here

Bohé Sec. II C inherits Taracchini (2014)'s mass fit: Barausse–Morozova–
Rezzolla Eq. (18), with its equal-mass limit replaced by Hemberger
Eqs. (9)–(10). A later primary publication resolves the unequal-mass spin
extension explicitly: **Ossokine et al., Phys. Rev. D 102, 044055 (2020),
Sec. III C, Eq. (3.5)**. It identifies this as the same fit used by previous
EOB models and explains that the expression had not previously been
written explicitly. In aligned variables, its equation simplifies to

```
a = X1^2*chi1 + X2^2*chi2
chi = a/(1-2*nu)
E_equal(chi) = 0.00258 - 0.0773/(chi - 1.6939)
M_final/M = 1 - nu*(1-4*nu)*(1-E_ISCO(a)) - 16*nu^2*E_equal(chi).
```

Here `E_ISCO(a)=sqrt(1-2/(3*r_ISCO(a)))` is the total specific Kerr ISCO
energy. Do not substitute a binding-energy value into that expression.
The spin argument in `E_equal` is `a/(1-2*nu)`, not `2*a` away from equal
masses. This later author equation makes the mass prescription actionable
without choosing an arbitrary extension of the equal-mass fit.

### Final spin: all coefficients available; selected block unresolved

Bohé names Hofmann–Barausse–Rezzolla (2016). Its aligned-spin Eq. (7) is

```
a_eff = a + xi*nu*(chi1+chi2)
a_final = a + nu*[L_ISCO(a_eff)-2*a*(E_ISCO(a_eff)-1)]
          + sum_(i=0..nM,j=0..nJ) k_ij*nu^(2+i)*a_eff^j.
```

Table 1 prints three alternatives: `(nM,nJ)=(1,2),(3,3),(3,4)` with
`xi=0.41616, 0.463926, 0.474046`, respectively. Eq. (11) fixes the omitted
`k00` from the equal-mass nonspin value 0.68646. All three are implementable.
Bohé's citation does not identify the adopted order, and the inspected
Ossokine remnant discussion likewise names Hofmann without selecting a
block. The exact author question is **which Table 1 block and numerical
precision were used by SEOBNRv4?** Choosing the largest block is a choice,
not evidence of the target's selection.

The current driver's Husa/PhenomD mass and spin fits are different
prescriptions. They cannot silently stand in for these inputs. Native
author-data QNMs provide a usable fundamental 220 mode for nonnegative
remnant spin within their documented range. The existing interpolant has
no negative-m branch; negative signed remnant spin requires appropriate
retrograde coverage or an explicit domain restriction. This is a further
coverage/integration task. In all cases use
`sigma_initial = (Re(Mf*omega)-i*decay_Mf)/(Mf/M)` exactly once.

## Concrete integration sequence after recording source choices

1. Reconstruct the inherited amplitude set through ell=8, including
   non-Keplerian velocity evaluated at zero radial momentum at the **same**
   radius/angular momentum. Feed these moduli, without NQC, into the
   published torque and existing circular/adiabatic initial conditions.
2. Integrate each system far enough to bracket the physical orbital-
   frequency maximum and calibrated offset, with trajectory refinement,
   explicit failure handling and numerical convergence checks. Keep the
   already documented conservative reconstruction choices visible.
3. Build complex `h22` using the declared complete phase and correct tail
   convention. Compute the base GW frequency as `-Im(hdot/h)` (or from a
   differentiable continuous phase), not a derivative of wrapped `angle`
   samples or simply `2*Omega`.
4. At `t_match=t_peak_Omega+Delta_t_peak_22`, select a converged local
   stencil, solve the existing NQC systems, and apply the correction to the
   waveform. Check all five target conditions and the actual local peak.
5. Compute the published mass and explicitly selected Hofmann spin, obtain
   the matching QNM branch, and evaluate the existing merger ansatz using
   the corrected matching amplitude, zero peak slope, frequency and phase.
   Join there; do not use the integrator's last sample as the match.
6. Reuse the driver's repaired strain scaling/projection/FFT machinery.
   Check physical limits, matching continuity, timestep/stencil convergence
   and gradients. These establish internal reconstruction quality;
   they do not replace missing target-convention evidence or prove parity.

This sequence is technically feasible. Calling its result the uniquely
intended published SEOBNRv4 path first requires the specific phase and
spin-block answers above. No additional isolated extension was implemented
as part of this assessment.

## Fresh primary sources and provenance

PDFs were retrieved on 2026-09-12 and read directly. Existing fresh primary
files from earlier passes supplied the direct waveform references listed
above. URLs and SHA-256 for this assessment's new retrievals:

| Source | PDF SHA-256 |
|---|---|
| [Bohé 1611.03703v3](https://arxiv.org/pdf/1611.03703v3) | `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba` |
| [Barausse et al. 1206.3803v3](https://arxiv.org/pdf/1206.3803v3) | `f4bc0155658529f791dccdbb2225dc574703d5aa03270c5df0af7fea9d15cc14` |
| [Hemberger et al. 1305.5991v2](https://arxiv.org/pdf/1305.5991v2) | `a3289b74619a5b84ed363c6ccc79864c6bfe1d23be75814e1d6d7674803d22e3` |
| [Hofmann et al. 1605.01938v2](https://arxiv.org/pdf/1605.01938v2) | `16bceff025a3084f5f424bd992b2659f92419da119349670ac0010d8ae80c8a3` |
| [Buonanno–Chen–Damour gr-qc/0508067](https://arxiv.org/pdf/gr-qc/0508067) | `c117f29e8bd3ceec2bf3048e5f455b7f67786f6960412869708ed7b2b1c8eb62` |
| [Ossokine et al. PRD 102, 044055, author-institute copy](https://pure.mpg.de/rest/items/item_3230468_5/component/file_3252882/content) | `a8dcecbc5269e68f8a121a431bc6759123ba4bee179caf15a94ddc07ee70f7b3` |

No NR or implementation mismatch was calculated for this audit. No code
was changed; the assessment file is its only repository write.
