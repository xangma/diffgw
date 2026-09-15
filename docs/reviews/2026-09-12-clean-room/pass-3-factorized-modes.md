# Factorized circular modes and flux, pass 3

`torchwave.approximants.eob.factorized_modes.FactorizedModes` reconstructs the
published Damour–Iyer–Nagar (DIN) circular nonspinning factorization through
ell=8. It also offers the leading finite-mass aligned-spin residual terms of
Pan et al. through ell=4. This is a separate building block, not a complete
SEOBNRv4 waveform or a claim of implementation compatibility.

Only primary mathematical publications were used. No LAL/PyCBC source,
implementation port, oracle output, residual fit, earlier implementation
coefficient specification, or author waveform program was consulted. SciPy's
complex Gamma is used solely in tests as an independent mathematical check.
This change owns the new module, focused tests, and this document; integration
into the existing waveform and radiation-reaction files is separate work.

## Primary sources

Equation numbers refer to these PDFs. Hashes are SHA-256 of the retrieved bytes.

| Publication | Use | PDF SHA-256 |
| --- | --- | --- |
| [Damour, Iyer and Nagar, arXiv:0811.2069v2](https://arxiv.org/pdf/0811.2069v2) | Flux (2), Newtonian factors (4)-(7), factorization/source (8)-(18), tail (19), residual phases (20)-(29), tail modulus (59), all rho coefficients (C1)-(C35) | `a60049d73b78f21fd0421ada236f9635c5023de8ab28a489aa3c1bd78857ba21` |
| [Pan et al., arXiv:1006.0431v2](https://arxiv.org/pdf/1006.0431v2) | Leading aligned-spin residual f terms (41a)-(41i); independent strain limits (38a)-(38e) | `4b51940aae4299c129b08d64f7f5ae92a5f9de00e96f2653abe719e76d596c19` |
| [Buonanno and Damour, arXiv:gr-qc/0001013v2](https://arxiv.org/pdf/gr-qc/0001013v2) | Independent total flux through 2PN, (3.31)-(3.33) | `c7746eb232e14fc4479ec2aec84152fb646a069956537d4e8d014954ccb87f7a` |
| [Bohé et al., arXiv:1611.03703v3](https://arxiv.org/pdf/1611.03703v3) | Scope check: v4 spin additions (2.4)-(2.6), NQC (2.7), flux (2.9), Appendix A peak targets; not inserted into this incomplete spin model | `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba` |

The corrected Pan v2 manuscript is used. Complex Gamma is evaluated from
[NIST DLMF 5.5.1](https://dlmf.nist.gov/5.5.E1) and
[5.11.1](https://dlmf.nist.gov/5.11.E1): shift the argument's real part to 16,
apply Stirling's expansion with eight exact-rational Bernoulli terms, and undo
the recurrence. The tail modulus uses DIN (59) directly with `expm1` for its
weak-field limit. No fitted Gamma coefficients are involved.

## Equations and normalization

Inputs have total mass M=1 and G=c=1:

- `state[...,4] = (r, phi, p_r_star, p_phi)`, with momenta per reduced mass.
- `omega = M Omega > 0` and `v = omega**(1/3)`.
- `effective_energy = H_eff/mu`, `real_energy = H_real/M`, both including rest mass.
- Mass fractions `x1+x2=1`, `nu=x1*x2`, and component spins `chi_i=S_i/m_i**2`.
- Outputs are `R h_lm/M`, with no distance or physical-mass conversion included.

The scalar harmonic in DIN (4) is evaluated at the equator with the standard
Condon–Shortley convention. The algebraic factorial expression preserves the
complex signs of the separately printed quadrupole/octupole strains. With
`epsilon=(ell+m)%2`, the Newtonian mass factor is
`c=x2**(ell+epsilon-1)+(-1)**m*x1**(ell+epsilon-1)`.

The source is `effective_energy` for even parity and `p_phi*v` for odd parity.
The latter is the J-source convention used throughout DIN Appendix C.
The complex tail uses `khat=m*omega*real_energy` and DIN's `r0=2M`, hence
the logarithm `log(4*m*omega)`. The nonspinning residual is `rho_lm**ell`.
Every coefficient of DIN (C1)-(C35) is included, at exactly its displayed
order. Their higher-order test-mass terms retain their published hybridized
meaning; they do not constitute additional finite-mass PN information.

All ten finite-mass residual phases in DIN (20)-(29) are retained. The paper
explicitly permits either `x` or `y` in its 2.5PN `y_bar` terms; this module
chooses `y_bar=x`. It uses `y**(3/2)=real_energy*omega` in the other terms.
DIN does not give finite-mass residual phases for modes above ell=4 except
(5,5). The default complex-mode policy raises for these modes. The explicit
`residual_phase="zero-unpublished"` choice omits only those unprovided residual
phases; known phases and the complex tail remain present. Amplitudes and flux
need no such choice because these real phase factors have unit modulus.

The optional `spin_model="pan2011-leading-f"` uses
`c*rho_nonspin**ell + c*f_spin`, where `f_spin` is precisely the linear-spin
term of Pan (41). This explicitly chosen additive prescription retains the
full DIN nonspinning series and only the published leading spin correction.
It does not exponentiate a spin-corrected rho, add unknown spin-squared terms,
or claim Pan's full higher-PN hybrid model. It is restricted to ell<=4; higher
modes do not silently retain nonspinning residuals under this option.

For odd m, products involving `chi_A/(x1-x2)` are evaluated using

```text
c/(x1-x2) = -sum(x1**j*x2**(p-1-j), j=0,...,p-1), p=ell+epsilon-1.
```

This exact divided-difference polynomial removes the apparent equal-mass
singularity and preserves spin-induced odd modes when the masses are equal
but the spins differ. Swapping both bodies gives `h_lm -> (-1)**m h_lm`.

`flux` uses the same mode amplitudes through
`sum_positive_m((m*omega)**2*abs(h_lm)**2)/(8*pi)`, including both m signs.
It is positive emitted energy per dimensionless time, not a force or an
angular-momentum derivative. The circular time derivative approximation is
`dot(h_lm)=-i*m*omega*h_lm`; explicit amplitude derivatives and eccentric
corrections are absent.

## API and explicit limits

```python
model = FactorizedModes()  # 35 positive-m nonspinning amplitudes through ell=8
amplitudes = model.amplitudes(state, omega, heff, hreal, x1, x2)
flux = model.flux(state, omega, heff, hreal, x1, x2)
h22 = model.mode(2, 2, state, omega, heff, hreal, x1, x2)

spin_model = FactorizedModes(4, spin_model="pan2011-leading-f")
spin_modes = spin_model(state, omega, heff, hreal, x1, x2, chi1=.4, chi2=-.2)
```

The selected `mode` method evaluates only the requested tail phase; the full
rho table is shared. It allows (2,2), (5,5), or any other published phase on
the ell=8 flux instance. Negative m uses `(-1)**ell*conj(h_lm)`. `forward`
retains its strict rejection if any requested phase is unprovided.

Float32/float64 state tensors and broadcastable scalar/tensor parameters are
accepted on the state's device. All arithmetic used in modes and flux remains
in Torch. Frequencies and energies must be positive; mass fractions must be
nonnegative and sum to one. Nonzero spins under the default nonspinning option
raise. Unknown model/phase choices raise rather than approximating full v4.

The state radius and radial momentum enter only through caller-supplied
frequency and energies. No non-Keplerian Newtonian velocity replacement is
made. This circular block alone does not justify extending the amplitude
prescription to arbitrary eccentric or plunging states.

The precise missing v4 ingredients are the inherited v2 finite-mass and
test-mass higher-order spin amplitude/phase terms, its complete residual
resummation prescriptions and velocity choices, and the v4 replacements in
Bohé (2.4)-(2.6). Adding only those three latter corrections would leave an
incomplete v4 base. No spin/NQC values have been inferred from oracle data.

For NQC, Bohé (2.7) requires three amplitude coefficients and two phase
coefficients. They are obtained from the dynamical trajectory by matching
amplitude, vanishing first amplitude derivative, amplitude curvature,
frequency, and frequency slope to Appendix A targets at the calibrated time
offset from the orbital-frequency peak. These trajectory-dependent solves,
their peak-time prescription, and merger attachment are absent here. Bohé
specifically excludes NQC from its v4 radiation-reaction flux and applies it
to the (2,2) waveform after the trajectory is computed. A local instantaneous
factorized-mode API cannot supply that calibration by itself.

## Verification

Eighteen focused tests passed:

- Five independently printed Newtonian complex strain limits, including signs.
- Newtonian, 1PN, 1.5PN tail and 2PN total flux coefficients at three mass ratios.
- Complex tail versus mathematical Gamma for every ell=2,...,8, from weak
  field to `omega=.2`; float64 relative tolerance `3e-14`.
- Agreement of all 35 mode moduli and their emitted flux; selected-mode phase
  policy and negative-m symmetry.
- Exchange symmetry, exact nonspinning equal-mass odd-mode zeros, and finite
  equal-mass unequal-spin 21/32 leading terms.
- Finite-difference checks of complex-mode and flux gradients in the state,
  frequency, both energies, mass ratio and both spins, including equal masses.
- High-mode flux derivatives, scalar/batch agreement, inference execution,
  and explicit rejection of unsupported physics and invalid input.

The 2PN check uses asymptotic extrapolation solely to isolate coefficients of
the calculated flux; it neither modifies the implementation nor provides
model coefficients. The PN source input follows DIN (10)-(18):
`H_eff/mu=1-x/2+(3/8+nu/6)*x**2` and
`p_phi*v=1+(3/2+nu/6)*x`, to the orders needed for this flux check.

```text
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_factorized_modes.py
18 passed
```

`git diff --check` passed. These checks establish the stated equations and
limits, not full SEOBNRv4 or merger-waveform accuracy.

## Experimental waveform integration

The parent subsequently added `FactorizedRadiationReaction` and explicit
`SEOBNR_ODE` selections `radiation_model="din2009_nonspinning"` and
`"pan2011_leading_spin"`. They use the 35-mode nonspinning flux and the
ell<=4 leading-spin flux, respectively, and supply the emitted (2,2) strain
from the same factorization. Radius and the chosen Hamiltonian supply both
rest-mass-inclusive energies at every dynamical state. The normalized torque
is `-flux/(nu*omega)`; the existing radial-force prescription is retained.
Neither selection changes the default `"prototype"` radiation model.

The initial-condition solve receives this same torque. Strain is multiplied
by `M_sec/D_sec` once, and the Fourier integration measure contributes its
separate time scale. The prototype ringdown now attaches to the complex
factorized endpoint when this option is selected. Its amplitude and phase
are continuous at zero attachment offset; no calibrated merger or NQC
behavior is thereby established. Flux includes multiple modes, while the
current `SEOBNR_ODE` output still projects only (2,2) and its symmetry partner.

The implementer independently reviewed these four integration files and
found no normalization or phase-sign defect. The parent directly checked
the Appendix C coefficient transcription against the cited primary PDF.
Eight new integration checks passed, including mechanical energy balance,
the Newtonian adiabatic inspiral limit, parameter derivatives, batch/inference
agreement, physical mass/distance/time covariance and rejection of unsupported
spins. The combined EOB component and waveform selection passed **67 tests**.
