# Published SEOBNRv4 (2,2) amplitude and phase-source audit

`seobnrv4_amplitude22.seobnrv4_published_amplitude22` implements a standalone
reconstruction of the published factorized **amplitude only**. It returns
`(R/M)|h22|` before NQC. It does not provide a complex mode, radiation flux,
trajectory, or merger attachment, and makes no implementation-compatibility
claim. Existing waveform defaults and radiation-reaction models are unchanged
by this component.

All implemented coefficients and formulae come from the primary PDFs below.
No waveform-output fitting or binary-oracle information enters this component.
Its only shared numerical helper is the independently reconstructed DIN tail
modulus in `factorized_modes.py`; no residual-amplitude table is imported.

## Sources inspected

Equation references below use the identified PDF versions. SHA-256 hashes
identify the bytes inspected.

| Primary publication | Relevant content | PDF SHA-256 |
| --- | --- | --- |
| [Taracchini et al., 1202.0790](https://arxiv.org/pdf/1202.0790) | A1–A6 Newtonian velocity, source and tail; A8a nonspin rho22; A16 phase | `d6d126be90bc789835ec363451f4ec51728a8d6ddc09eff078620fdc46860745` |
| [Pan et al., 1006.0431v2](https://arxiv.org/pdf/1006.0431v2) | 26 tail convention; 27a Kerr phase; 29a Kerr residual amplitude; 38a strain; 42a leading comparable-mass spin amplitude | `4b51940aae4299c129b08d64f7f5ae92a5f9de00e96f2653abe719e76d596c19` |
| [Buonanno, Faye and Hinderer, 1209.6349v2](https://arxiv.org/pdf/1209.6349v2) | 3.4 spin combination; 4.26–4.27b comparable-mass 2PN spin-squared h22 | `1252291e97215e07fb92a028d8faffb6bcd95122b895312c6692a993fd318af0` |
| [Taracchini et al., 1311.2544](https://arxiv.org/pdf/1311.2544) | Eq. 4 and surrounding v2 text: 2PN comparable-spin terms, Kerr hybridization, removal of calibrated rho22 term, calibrated phase addition | `f57b099e701c1206802d9887f6843ce75fc8753ac9c3ae5c2137933160f85967` |
| [Bohé et al., 1611.03703v3](https://arxiv.org/pdf/1611.03703v3) | 2.4–2.5 amplitude replacements; 2.1 and 2.6 phase-energy ambiguity | `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba` |
| [Damour, Iyer and Nagar, 0811.2069v2](https://arxiv.org/pdf/0811.2069v2) | 59 exact tail modulus | `a60049d73b78f21fd0421ada236f9635c5023de8ab28a489aa3c1bd78857ba21` |
| [Cotesta et al., 1803.10701v2](https://arxiv.org/pdf/1803.10701v2) | IV B–C and Appendix A: 22 is unchanged from v4; no consolidated 22 phase supplied | `0d12a98834565ace51327ddfc7b006c3cd48389421a6e84b2fd9372f96c1664d` |

## Normalization and non-Keplerian prefactor

Set G=c=M=1 internally. Inputs are:

- `omega = M Omega`, for the actual dynamical state;
- `omega_zero_radial = partial(Hreal/mu)/partial(p_phi)` evaluated at
  `p_r=0`, keeping the actual state's r and p_phi fixed;
- `heff_over_mu = Heff/mu`;
- `hreal_over_M = sqrt(1+2 nu (Heff/mu-1))`, **including rest mass**;
- `x1=m1/M`, `x2=m2/M`, and individual aligned dimensionless spins.

The caller supplies these quantities. In particular, `omega_zero_radial`
does not mean a newly solved circular-orbit frequency. Zeroing p_r_star is
equivalent to zeroing p_r where the tortoise transformation is nonsingular.
An additive constant in Hreal does not affect that derivative.

Taracchini A2 prescribes

```
v = omega^(1/3)
v_phi = omega * omega_zero_radial^(-2/3)
khat = 2 * omega * hreal_over_M
|T22|^2 = 4*pi*khat/(1-exp(-4*pi*khat)) * (1+4*khat^2)*(1+khat^2)
(R/M)|h22| = 8*sqrt(pi/5)*nu*v_phi^2*(Heff/mu)*|T22|*rho22^2.
```

Only the Newtonian factor uses v_phi; the residual PN series uses v.
For zero radial momentum the two supplied frequencies coincide and
v_phi=v exactly. Pan 26 and Taracchini A6 specify r0=2M/sqrt(e), unlike
DIN's r0=2M, but this difference affects only the discarded tail phase.
The tail modulus uses total rest energy, consistent with Pan's stated
test-particle limit Hreal=M. Multiplication by GM/(c²R) gives physical
strain amplitude. Both frequencies must be dimensionless rather than Hz.

## Consolidated amplitude prescription

Write n=nu=x1*x2, d=x1-x2, S=(chi1+chi2)/2, A=(chi1-chi2)/2,
qK=(1-2n)S+dA, q0=S+dA, and L=EulerGamma+log(4v). The implemented
series is `rho22=1+sum(c_k*v^k)` for k=2,3,4,5,6,7,8,10:

```
c2 = 55n/84 - 43/42
c3 = -2[(1-n)S+dA]/3
c4 = 19583n²/42336 - 33025n/21168 - 20555/10584 + q0²/2
c5 = (-34/21+49n/18+209n²/126)S - (34/21+19n/42)dA
c6 = 10620745n³/39118464 - 6292061n²/3259872 + 41pi²n/192
     -48993925n/9779616 - 428L/105 + 1556919113/122245200 + 89qK²/252
c7 = (18733/15876+74749n/5292-245717n²/63504+50803n³/63504)S
     +(18733/15876+50140n/3969+97865n²/63504)dA
c8 = 9202L/2205 - 387216563023/160190110080 + 18353qK²/21168 - qK⁴/8
c10 = 439877L/55566 - 16094530514677/533967033600.
```

The nonspin coefficients are Taracchini A8a, with its calibrated
`nu*rho22^(4)*v^8` term removed as explicitly prescribed in the v2 paper.
c3 is Pan 42a. For black holes, Buonanno 3.4 gives
`S0+/M²=S0-/M²=x1*chi1+x2*chi2=q0`; 4.26–4.27b therefore supplies
`q0²*v^4` in the normalized strain. No other factor contributes a
spin-squared term at that relative order on circular orbits, giving
`q0²/2` in rho22. This replaces Pan's test-mass `qK²/2` coefficient.

The v2 text prescribes the remaining Kerr spin terms of Pan 29a with
q=qK. Bohé 2.4–2.5 replaces the spinning test-particle terms at v^5 and
v^7. This reconstruction follows that stated replacement literally:
**the entire old v^7 spin coefficient is replaced**, including Pan's
`qK³/3`. It does not append that cubic term to Bohé 2.5. This explicit
reading is part of the source boundary, not a claim about another
implementation. The qK² v^6 and qK²/qK⁴ v^8 terms remain as published
test-mass hybrid terms; they are not newly known comparable-mass results.

## Why no complex mode is supplied

The inspected primary chain does not uniquely settle the complete residual
phase to the same standard. The unresolved information is specific:

1. Taracchini 2012 A16 gives the nonspin phase, while the v2 paper adds
   `540*nu*chi*v^6` for chi>=0, where
   `chi=S+d*A/(1-2nu)`. Bohé adopts the earlier waveform expressions and
   describes new phase corrections, but does not explicitly state whether
   this earlier calibrated phase term remains. Literal inheritance suggests
   retaining it; an unambiguous consolidated v4 expression or explicit
   deletion/retention statement is missing from the inspected chain.
2. The v2 text also describes importing known test-particle spin effects.
   Pan 27a's phase includes `-4*q*v^6/3+20*q*v^8/63`. It is unclear whether
   that general v2 statement imports these phase terms as well as the
   residual amplitudes, and, if it does, exactly how they combine with
   Bohé's new 2.6 term. Substitution q=qK already supplies the same
   leading test-mass phase term. Blindly adding 2.6 would double count it.
   The status of the `20*qK*v^8/63` term also needs explicit confirmation.
3. Bohé 2.1 defines HEOB with rest mass subtracted, but 2.6 prints
   `-4*qK*(Omega*HEOB)^2/3` and calls it a 3PN correction. With binding
   energy, this scales as v^10 rather than v^6 and vanishes in the
   test-mass limit, contradicting Pan 27a. Total rest energy repairs both
   limits. This is a physically supported interpretation, but the printed
   symbol is inconsistent; the same distinction matters for the inherited
   tail/phase formulae. It is stated explicitly for this amplitude API.

Cotesta's subsequent v4HM paper states that the 22 mode is unchanged from
v4, and focuses its explicit residual expressions on higher modes. It
does not resolve these 22 inheritance questions. A primary consolidated
delta22 expression, author clarification, or erratum specifying those
choices would permit a complete complex-mode reconstruction. The
amplitude does not depend on any of them.

## Validation

`python -m pytest tests/test_clean_room_seobnrv4_amplitude22.py -q`:
**20 passed**. Checks cover independently published full-strain Newtonian,
1PN and 1.5PN spin coefficients (Pan 38a), the 2PN spin-squared strain
coefficient (Buonanno 4.27b), isolation of Bohé's v^5 and v^7 spin
coefficients, Pan's remaining Kerr spin powers, absence of the replaced
cubic-spin term, the zero-radial circular limit, and the exact
`omega_zero_radial^(-4/3)` amplitude scaling.

The tail is separately checked against SciPy's complex Gamma function.
Torch gradcheck covers both frequencies, both energies, the independent
mass fraction and both spins, including equal masses. Broadcasting,
body exchange, the zero-mass endpoint, float32 inference and invalid
inputs are covered. Weak-field coefficient limits use Richardson
extrapolation only in tests, with no fitted correction to the model.

A second fresh-context implementer independently checked the seven primary
PDFs, their recorded hashes, the non-Keplerian normalization and the stated
coefficient prescription. It found no actionable defect within this explicitly
limited amplitude API and reproduced all 20 focused tests. This review does
not settle the residual-phase ambiguities listed above.
