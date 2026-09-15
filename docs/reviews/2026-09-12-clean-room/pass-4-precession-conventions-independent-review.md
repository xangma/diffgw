# Independent review of the published precession conventions

The bounded review found three defects during development; all three are
resolved in the reviewed implementation. No remaining actionable equation,
phase-plumbing, or scalar/batch defect was found within this scope. The reviewer
made no implementation or test changes.

## Scope and evidence

The review covered raw/reference-relative NNLO and MSA angle selection,
Pratten Table IV offsets and the optional carrier phase, the XP/XPHM wrapper
calls, observer geometry, aligned limits, and the focused convention tests.
The mathematical references were the freshly retrieved
[Pratten et al. 2004.06503v2](https://arxiv.org/pdf/2004.06503v2)
(Table III/IV, Eq. 4.4, Appendix C and the Wigner rotation) and
[Bohé et al. T1500602-v4](https://dcc.ligo.org/public/0122/T1500602/004/PhenomPv2_technicalnotes.pdf)
(Eqs. 22–25 and 37–40). Their PDF SHA-256 hashes are respectively
`fd21fcd7c111d780eddb109e51c5813d7645a52022b84a24d923056b639bd935`
and `5847dd72a26b25eb75d3913c474e65cd0e6025a1637f95424b9ed01cfd033f02`.
No external waveform implementation, waveform oracle, historical calibration
specification, or residual fitting was used.

The five Table IV rows, `gamma=-epsilon`, and each carrier orbital-phase
argument agree with the printed table. The observer-to-J Cartesian Pv2 alpha
agrees modulo 2pi with the XP construction when both use the same J and source
geometry. Source-frame N and the observer polarization vector are Eq. C4 and
C24; the inverse polarization rotation agrees with D6–D7.

## Resolved findings

1. **Observer azimuth was omitted from the analytical precessing branch.**
   XPHM's exactly aligned branch accepted `phi0`, producing a discontinuity
   from arbitrarily small transverse spin. The fix uses `phiRef-phi0` in N/P
   geometry while retaining the specified Table IV carrier argument. XP's
   aligned carrier now receives the corresponding observer-azimuth phase.
2. **Reference frequency alone could not create a parameter batch.**
   Including `f_ref` in the common parameter broadcast fixes the shape error
   and preserves its gradients.
3. **The opposed-J aligned limit exposed an unsupported NNLO expansion.**
   At beta=0 the surviving Wigner phase is `exp[i*m*(alpha-epsilon)]`; at
   beta=pi it is `exp[-i*m*(alpha+epsilon)]`. The zero-cone NNLO coefficients
   satisfy `alpha_dyn=epsilon_dyn`, so the latter retains a frequency-dependent
   phase. Constant Table IV offsets cannot remove it. The fix explicitly
   rejects NNLO/fallback precessing rotations with `L+S_parallel<=0` at the
   reference or any evaluated mode frequency. The separate exactly aligned
   default carrier remains available, including in mixed batches.

The last change is a conservative support boundary. T1500602 Eq. 24 chooses
positive cos(beta) during flips and states that their accuracy is uncontrolled;
Pratten Eq. 4.4 gives a signed cosine and excludes transitional precession from
the fixed-J approximation. Those sources do not establish an additional signed
NNLO phase repair. The code makes no continuity claim across the rejected
domain, and it does not reject valid MSA rows solely for a cone above pi/2.

## Independent regression probes

An independent temporary Python probe, without adding repository tests, passed:

- XP and individual XPHM 21/22/32/33/44 carriers at `phi0=0.4`, approached with
  transverse spin magnitude `1e-10` from three azimuths; the maximum relative
  polarization error against the exactly aligned carrier was `6.6e-10`.
- XP and XPHM33 with `f_ref=[30,40]` as the only batched input, all five
  conventions: agreement with scalar calls and finite autograd derivatives.
- Opposed-J near-aligned inputs (`m1=100`, `m2=5`, `chi1z=-0.95`) rejected for
  all five conventions in XP and XPHM33. Mixed batches preserved the exactly
  aligned convention-1 row and agreed with independent scalar calls.

The reviewer also read the author's 51 focused convention tests, including
finite-difference gradients, MSA cases, reference/per-mode domain checks, and
valid MSA cones above pi/2. The author runs the focused/combined suites; their
counts are recorded in the companion implementation report. The four source
files passed `git diff --check`.

## Limits of the conclusion

This verifies the stated Table IV mathematical conventions and their native
plumbing. It does not establish an external implementation's free additive
gauge for unanchored MSA primitives in options 5/7, nor the undocumented
distinctions among MSA version labels. The selected version supplies one J;
the papers do not unambiguously require a second 2PN J for Pv2 alpha alone.

The public wrappers retain their existing default `precession_method='nnlo'`,
which selects 102. Choosing `precession_method='msa'` selects 223; Table III
lists 223 as the paper implementation's default. This pass changes neither
native defaults nor resolves that target-default difference. Full target-model
equivalence, carrier calibration, remnant prescriptions, and general
experimental ODE observer geometry remain outside this review.

Reviewed implementation SHA-256 snapshots:

| File | SHA-256 |
|---|---|
| `imrphenomp.py` | `9647e7d1db998c53d57a01f7ade4933a630945e3244bd20e60384a7d3a0f2447` |
| `precession_msa.py` | `5e0d0b09af6f16a050e6f9eddb8f4b66c754b03634e39a43eb44461ebdf06842` |
| `imrphenom_xp.py` | `cc15e18f07221c9106e425abfbf3257a71b3c6f77e2056f0e9fdeffe848bcbe5` |
| `imrphenom_xphm.py` | `bb591b531dd1be8c781386a6fc0792ad954898a0b7f0825f62838a09de239ef8` |
| `test_clean_room_precession_conventions.py` | `2a894fe86101171d7b8471d2c7e8303d06c881db90ace1203f00ab6049ab59e1` |
