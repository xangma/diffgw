# Published factorized (2,2) phase reconstruction

Date: 2026-09-12.

`src/torchwave/approximants/eob/seobnrv4_mode22.py` supplies the complex
factorized (2,2) mode before NQC, with a **required caller-supplied residual
phase**. Separate helpers implement the printed phase components. An
optional composer requires all unresolved spin selections explicitly.
This is an experimental published reconstruction, not a uniquely identified
SEOBNRv4 prescription or an implementation-parity claim.

Only this new source module, its new focused test and this report were
written. The shared waveform driver, integrator, amplitudes, flux and
merger attachment were not changed.

## Source evidence

The authorized integration assessment and amplitude report were read for
scope. The primary papers themselves were then opened from arXiv and their
equations checked against the local primary PDFs. No LAL/PyCBC source,
source port, oracle output, target implementation or mismatch was inspected.
No residual informed a phase choice or coefficient.

| Primary source | Direct evidence | PDF SHA-256 |
| --- | --- | --- |
| [Taracchini et al., 1202.0790v2](https://arxiv.org/pdf/1202.0790v2) | A1–A3: Newtonian sign and orbital phase; A5–A6: source and complex tail; A16: full nonspin residual phase; Eq. 2: binding-energy notation conflict | `d6d126be90bc789835ec363451f4ec51728a8d6ddc09eff078620fdc46860745` |
| [Pan et al., 1006.0431v2](https://arxiv.org/pdf/1006.0431v2) | 21: phase logarithm scale; 26: tail radius and explicit total-energy test-mass limit; 27a: Kerr residual phase; 38a: independent complex full-strain weak-field limit | `4b51940aae4299c129b08d64f7f5ae92a5f9de00e96f2653abe719e76d596c19` |
| [Taracchini et al., 1311.2544](https://arxiv.org/pdf/1311.2544) | Eq. 4 and following text: factorization, test-particle spin hybridization and positive-chi phase calibration | `f57b099e701c1206802d9887f6843ce75fc8753ac9c3ae5c2137933160f85967` |
| [Bohé et al., 1611.03703v3](https://arxiv.org/pdf/1611.03703v3) | 2.1: binding-energy definition; 2.6: stated 3PN spin phase; surrounding text: inheritance and explicitly stated amplitude replacements | `db9dc054368f131bdc152b319e54441c665547668221891723b309b005ac23ba` |

The shared numerical Gamma phase was checked against the mathematical
recurrence in [DLMF 5.5.1](https://dlmf.nist.gov/5.5.E1) and the
Stirling expansion in [DLMF 5.11.1](https://dlmf.nist.gov/5.11.E1).
These establish its numerical algorithm, not any waveform coefficient.

## Unambiguous equations and declared energy repair

Use dimensionless quantities with G=c=M=1. Define

```
omega = M*Omega
v = omega^(1/3)
E = Hreal_total/M = sqrt(1+2*nu*(Heff/mu-1))
u = omega*E
khat = 2*u
qK = (1-2*nu)*chiS + (x1-x2)*chiA
chi = qK/(1-2*nu).
```

Both Taracchini A6 and Pan 26 specify `r0/M=2/sqrt(e)`. The complete
complex tail is therefore

```
T22 = Gamma(3-2i*khat)/2 * exp(pi*khat)
      * exp(2i*khat*log(8*omega/sqrt(e))).
```

`tail22_phase` evaluates its continuous logarithmic phase. The inherited
DIN numerical helper uses radius 2M; the required conversion is exactly
`phase_Taracchini = phase_DIN - khat`. This is a constant radius change
in the printed logarithm. The Gamma recurrence is unchanged. `tail22`
combines that phase with the independently reconstructed exact tail modulus.

Taracchini A16 gives the following full nonspin-coefficient phase:

```
delta_NS = 7*u/3 + 428*pi*u^2/105
           + (1712*pi^2/315-2203/81)*u^3 - 24*nu*v^5.
```

The `u^3` term is included. Its test-particle limit is Pan 27a's printed
nonspin `v^9` term. The phase can depend on spin through the supplied total
energy, despite the name referring to its nonspin PN coefficients.

The total energy in these expressions is an explicit repair of inconsistent
symbols. Taracchini Eq. 2, Taracchini 2014 Eq. 1 and Bohé Eq. 2.1 define
their real/EOB Hamiltonian with the rest mass subtracted. A literal binding
energy in the tail would contradict Pan 26, which explicitly gives
`Hreal -> M` in the test-particle limit. It would also remove the leading
`v^3` residual phase and turn Bohé's stated 3PN correction into a `v^10`
term. Using total real energy restores these independent limits. The API
names this quantity `hreal_over_M` and explicitly documents rest mass.
It does not claim the printed binding-energy and total-energy symbols
are literally identical.

The separate spin components are

```
delta_Bohe  = -4*qK*u^2/3
delta_Pan6  = -4*qK*v^6/3
delta_Pan8  = 20*qK*v^8/63
delta_540   = 540*nu*max(chi,0)*v^6.
```

The first line is Bohé 2.6 with the total-energy repair just described.
Pan 27a supplies the next two lines, with the explicitly declared Kerr
hybridization `q -> qK` from the v2 discussion. The last line is the
calibration printed after Taracchini 2014 Eq. 4. Together with complete
body relabeling, `chi=qK/(1-2nu)` reproduces its ordered-mass expression
and remains regular at equal masses. Its derivative has a physical kink
at `chi=0`; autograd takes the nonnegative branch's one-sided derivative.

## Required choices and public API

The primary chain does not uniquely specify retention of the 540 term,
whether the Kerr phase was inherited along with the amplitude, or how
its 3PN term combines with Bohé 2.6. Those questions remain open.

`seobnrv4_published_delta22` requires these keyword arguments without defaults:

| Argument | Explicit choices |
| --- | --- |
| `spin_v6` | `bohe_total_energy`, `pan_test_particle`, `bohe_plus_pan`, `none` |
| `include_pan_v8` | `True` or `False` |
| `include_positive_chi_540` | `True` or `False` |

Every combination includes A16. The explicit sum remains available as a
deliberate inheritance reading. At `E=1`, Bohé and Pan's 3PN terms are
equal, so their sum doubles that leading Kerr coefficient; this mathematical
fact does not identify which inheritance combination the target used.
No choice is advertised as its default. Individual public component helpers
also permit a caller to construct a separately justified phase expression.

`seobnrv4_published_mode22(omega, phi, *, residual_phase, ...)` reuses the
provided published amplitude and requires the caller's `delta22`. The
remaining keyword arguments are the amplitude module's existing inputs.
Its output is dimensionless `(R/M)*h22`, before NQC, with the phase
convention

```
h22 = -A*exp(i*(-2*phi + arg_continuous(T22) + delta22))
    =  A*exp(-i*phase22)
phase22 = 2*phi - arg_continuous(T22) - delta22 + pi.
```

The negative Newtonian sign follows from Taracchini A1/A3 and is independently
printed in Pan 38a. `seobnrv4_published_phase22` returns the continuous
`phase22` above, including its constant pi. A caller can differentiate it
along its trajectory to obtain `M*omega_GW = d(phase22)/d(t/M)`. This
frequency includes evolving tail and residual phases. The helper does not
replace it by `2*omega` or differentiate a wrapped complex argument.
The supplied orbital and residual phases must themselves be continuous.

Inputs broadcast using omega's device and float32/float64 dtype. Outputs
are complex64/complex128 for the mode and real for the phase. Frequencies
and energies must be finite and positive; the amplitude's mass and spin
domain checks remain in force. The supplied positive-frequency convention
does not cover a retrograde signed orbital-frequency parameterization.
The amplitude's existing replacement/hybridization interpretation also
remains part of this reconstruction.

## Verification

Command:

```
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest \
  tests/test_clean_room_seobnrv4_mode22.py \
  tests/test_clean_room_seobnrv4_amplitude22.py -q
```

**48 passed**, including 28 new phase/mode tests and the 20 existing
amplitude checks. The new checks include:

- Complex tail and unwrapped phase against independent SciPy log-Gamma,
  over `omega=1e-12` through `2` and three energies, in both floating
  dtypes. This extended numerical interval tests the helper rather than
  claiming physical waveform validity there.
- Tail derivatives against independent SciPy digamma expressions.
- Full complex waveform Newtonian sign and Pan 38a's imaginary 1.5PN
  strain coefficient, using Pan 21's independently printed logarithm scale.
- The nonspin test-particle `v^9` coefficient, the repaired Bohé PN order
  and its equality to the Pan 3PN term at total energy E=1.
- Required independent choices, deliberate-sum availability and the
  calibrated term's piecewise derivative.
- Continuous phase derivative against `-Im(hdot/h)` on a changing
  trajectory across argument wraps; it differs from `2*omega` as required.
- Mode modulus, orbital-phase covariance, full body exchange,
  scalar-versus-batch agreement, inference mode and the zero-mass endpoint.
- Complex end-to-end gradcheck through both frequencies, both energies,
  independent mass fraction, both spins and orbital phase, including
  equal masses; invalid-input rejection.

These are equation, numerical and internal-consistency checks. They do
not establish a missing phase-selection rule or binary parity.
