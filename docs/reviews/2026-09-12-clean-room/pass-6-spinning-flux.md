# Published spinning amplitude moduli and radiation reaction

The new `SEOBNRv4PublishedFlux` reconstructs the inherited spinning amplitude
recipe through all 35 positive-m modes with 2 ≤ ell ≤ 8. It provides moduli,
energy flux, and a Hamiltonian adapter. This is an explicit published-equation
reconstruction; neither selectable truncation establishes implementation parity.
Residual phases and NQC factors are absent. Bohé Eq. (2.9) only needs squared
moduli, and the accompanying text excludes NQC from the v4 radiation reaction.

The implementation is in `src/torchwave/approximants/eob/seobnrv4_flux.py`.
The earlier reviewed DIN prefactors, nonspinning residuals and tail modulus,
Pan leading-spin regularization, and Bohé 22 amplitude are reused. The only
changes to existing implementation code are algebraically equivalent squares
in `factorized_modes.py`, necessary to preserve flux Hessians at vanishing modes.
No waveform driver is selected by these changes.

## Primary sources

The PDFs were obtained directly from primary publications during this audit.
No external waveform implementation, oracle result, historical calibration
specification or inferred calibration table was used.

| Source | Equations used | PDF SHA-256 |
|---|---|---|
| [Damour, Iyer and Nagar, 0811.2069](https://arxiv.org/abs/0811.2069) | (4)–(6), (59), (C1)–(C35) | `a60049d73b78f21fd0421ada236f9635c5023de8ab28a489aa3c1bd78857ba21` |
| [Pan et al., 1006.0431v2](https://arxiv.org/abs/1006.0431v2), including appended erratum | (28), (29), (33), (38), (40)–(42), (D1), (F26) | `4b51940aae4299c129b08d64f7f5ae92a5f9de00e96f2653abe719e76d596c19` |
| [Taracchini et al., 1202.0790](https://arxiv.org/abs/1202.0790) | (A1)–(A15) | `d6d126be90bc789835ec363451f4ec51728a8d6ddc09eff078620fdc46860745` |
| [Taracchini et al., 1311.2544](https://arxiv.org/abs/1311.2544) | (4) and surrounding amplitude prescription | `f57b099e701c1206802d9887f6843ce75fc8753ac9c3ae5c2137933160f85967` |
| [Buonanno, Faye and Hinderer, 1209.6349v2](https://arxiv.org/abs/1209.6349v2), BFH | (3.4), (4.27) | `1252291e97215e07fb92a028d8faffb6bcd95122b895312c6692a993fd318af0` |
| [Bohé et al., 1611.03703v3](https://arxiv.org/abs/1611.03703v3) | (2.4), (2.5), (2.9) | `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba` |

## Construction and source choices

Write δ = X1 − X2, ν = X1 X2, χS = (χ1+χ2)/2 and χA = (χ1−χ2)/2.
Taracchini 2014 specifies the Kerr-import spin
q = χKerr = (1−2ν)χS + δχA. Its comparable-mass terms replace the corresponding
test-particle coefficients, following Pan's hybridization prescription.

The nonspinning base is DIN Appendix C with nine additional test-particle
terms from Taracchini Appendix A: 21 at v¹⁰; 32 at v⁸; 43 and 41 at v⁶;
54, 52, 66, 64 and 62 at v⁴. The calibrated νρ22^(4) v⁸ term is removed,
as specified in Taracchini 2014. Taracchini (A11a) prints an extra minus before
the rational ρ55 v² coefficient. The implementation retains DIN's sign,
which also gives Pan (D1a)'s required ν=0 coefficient −487/390.

For odd m with ell ≤ 4 the residual is ρNS^ell + fS. It uses the ordinary
f series in Pan (28), regularized finite-mass leading terms in (41), and the
BFH 21 correction below. Other modes use ordinary ρ from Pan (29)/(D1).
Pan's alternative factorization ρᶠ in (35) is a different construction and
is not substituted for these ordinary residuals.

| Modes | Spin terms retained in their residual |
|---|---|
| 22 | Reviewed Bohé amplitude: comparable-mass 1.5PN and 2PN terms; whole v⁵ and v⁷ spin-coefficient replacements from Bohé; remaining printed Kerr terms |
| 21 | Finite-mass f at v and v³; Kerr f at v⁴, v⁵, v⁶, v⁷ from Pan (28b) |
| 33, 31 | Finite-mass f at v³; Kerr f at v⁴, v⁵, v⁶, v⁷ from Pan (28c,e) |
| 43, 41 | Finite-mass f at v; Kerr f at v⁴, v⁵ from Pan (28g,i), including the printed cubic-spin term in f41 |
| 32 | Finite-mass ρ at v and the induced v² term from (F26); Kerr ρ at v³ through v⁶ from (29d) |
| 44, 42 | Finite-mass ρ at v³; Kerr ρ at v⁴, v⁵, v⁶ from (29f,h) |
| 55, 53, 51 | Kerr ρ at v³, v⁴, v⁵ from (D1a,c,e) |
| 54, 52 | Kerr ρ at v³, v⁴ from (D1b,d) |
| 66, 64, 62 | Kerr ρ at v³, v⁴ from (D1f–h) |
| 65, 63, 61; 77, 75, 73, 71 | Kerr ρ at v³ from (D1f–k) |
| 76, 74, 72; 87, 85, 83, 81 | No spin coefficient occurs within the selected printed truncations |
| 88, 86, 84, 82 | Required explicit selection described below |

Pan's (D1l,m) tabulates ell=8 only through v², but its general even-parity
formula (33) supplies −2qv³/3. Taracchini's inheritance wording does not
explicitly settle whether to extend those tabulated rows. Therefore the
constructor requires `spin_truncation="tabulated"` or
`spin_truncation="generic-leading-ell8"`. The second includes this extra v³
coefficient for the four even-parity ell=8 modes; all other modes coincide.
The option is mandatory even for a smaller `lmax`, so the source convention
remains explicit when a caller later requests the full flux.

## Two finite-mass terms that require care

For 21, BFH (4.27a), transformed to Pan's mode-phase convention, gives the
normalized spin coefficients

    h21_spin / h21_Newtonian = b1 v + b3 v³,
    b1 = −3(χS + χA/δ)/2,
    b3 = [(-7+33ν)χS + (-7+205ν)χA/δ]/14.

BFH uses Sc/M² = (1−2ν)χS+δχA and Σc/M² = −δχS−χA.
Pan (40)'s normalized angular-momentum source is

    j v = 1 + j2 v² + j3 v³,
    j2 = (9+ν)/6,
    j3 = −10[(1−ν/2)χS + δχA]/3.

Dividing by that source gives f1=b1 and f3=b3−j2 f1−j3, hence

    f21_spin = −3(χS+χA/δ)v/2
             + [(427+79ν)χS + (427+131ν)χA/δ]v³/84
             + the remaining printed Kerr terms.

At ν=0 its v³ term is 61q/12, matching Pan (28b). It replaces that
test-particle coefficient. The apparent δ denominators are never evaluated:
for odd m, c = X2^p−X1^p with p=ell+epsilon−1, and
c/δ = −sum[k=0..p−1] X1^k X2^(p−1−k). Thus equal masses with unequal spins
remain finite, and component exchange preserves the moduli.

For 32, Pan (F26) requires a v² spin term −a1², where
a1 = −4νχS/[3(3ν−1)]. In (1+a1 v−a1² v²)^3 the spin-square v²
coefficient cancels. Keeping only the leading ρ32 v spin term would generate
a spurious spin-square contribution at the absolute 2PN order of this mode.

## Units, adapters and derivatives

State entries are (r, φ, p_r*, p_φ), using M=G=c=1 and momenta divided by μ.
Amplitudes are |Rh_lm/M|. Frequencies are MΩ. The inputs Heff/μ and Hreal/M
include rest mass; the latter is sqrt(1+2ν(Heff/μ−1)).

The Newtonian prefactor uses vφ=Ω Ω0^(−2/3), where Ω0 is evaluated at the
same r,pφ with zero radial momentum. It is not obtained by solving for a
different circular angular momentum. Residuals use v=Ω^(1/3); the source
is Heff/μ for even parity and pφ v for odd parity. The tail modulus uses
khat=mΩHreal/M. Taracchini's tail radius changes the phase, so it does not
enter this modulus calculation.

`from_hamiltonian(state, hamiltonian, omega=None)` computes both frequencies
and energies using the existing inference-safe local Hamiltonian partials.
An explicitly supplied Ω is retained. Parameter batches preserve the native
Hamiltonian convention of appending trajectory dimensions.
`radiation_reaction_forces(omega, pr_star, pphi, *, r, hamiltonian)` returns

    flux = sum[m>0] (m Ω)² |Rh_lm/M|² / (8π),
    Fφ = −flux/(ν Ω),       Fr* = (pr*/pφ) Fφ.

The factor accounts for both signs of m once. Exactly ν=0 is valid for
amplitude/flux evaluation and gives zero flux; reduced-mass forces require
ν>0. Inputs require positive radius, pφ, frequencies and energies. Float32,
float64, broadcast batches and autograd are supported.

Amplitude moduli have cusps at their zeros, but energy flux is smooth there.
Flux therefore uses the signed real residual and direct squares. The shared
`energy_flux_from_modes` similarly uses real²+imaginary² for complex modes
and a direct square for real inputs. `FactorizedModes.flux` now supplies its
existing prefactors before taking an absolute value. This fixes the equal-mass
odd-mode Hessian without changing physical flux values or selecting a phase.

## Verification

The focused command passed **72 tests**:

    python -m pytest tests/test_clean_room_factorized_modes.py \
      tests/test_clean_room_seobnrv4_flux.py \
      tests/test_clean_room_eob_factorized_integration.py \
      tests/test_clean_room_seobnrv4_amplitude22.py -q

Checks cover all-mode Newtonian limits and non-Keplerian powers; independent
complex-Gamma tail ratios; the BFH h21 coefficient recovered after dividing
out the PN angular-momentum source; the 32 spin-square cancellation; equal-mass
continuity and exchange symmetry; all-mode flux first and second derivatives
through the Hamiltonian; batching; inference-created tensors; float32 accuracy;
and input failures. A separate physical h21+h33 weak-field test verifies the
nonzero equal-mass flux curvature directly from Pan (38b,c), for both complex
modes and their real/imaginary components. The public DIN flux Hessian is also
checked at equal masses. No waveform implementation serves as an oracle.
