# Pass 22: clean-room EOB horizon and convention diagnosis

Date: 2026-09-13

The public-binary diagnostic case is the frozen 30 + 20 solar-mass,
zero-spin case in the pass-15 verification catalog (`SimInspiralFD`,
`SEOBNRv4`, `f_low=20 Hz`, `f_high=256 Hz`, `df=0.5 Hz`, oracle upper
frequency 1024 Hz). All TorchWave runs used float64 CPU tensors. No
LALSimulation or PyCBC implementation source was inspected.

The existing prototype dispatch gives complex relative L2 errors of
1.0206987416256426 (plus) and 1.0204101872808298 (cross). Its magnitude
relative L2 errors are 0.8368897308954389 and 0.8397626858863945; the flat
mismatches are 0.966318375826886 and 0.9661912122149872. Optimizing one
constant complex phase changes the L2 values only to 1.0205871789836685 and
1.0201927517965719. Therefore a constant phase/sign convention cannot account
for this residual; the magnitude error is also too large for a time gauge.

The published driver already defines a 20,000-step default, while the native
`SEOBNR_ODE` wrapper historically defined a 1,500-step prototype default and
silently forwarded it to both assemblies. With `dt_geom=1` and 6,500 steps,
the published equations reach the first resolved orbital-frequency maximum at
`t/M = 5002.714605426067` and complete NQC/C1 attachment. The wrapper now
selects 20,000 steps when `num_steps` is omitted for
`assembly_model='seobnrv4_published'`, preserving the published driver's
existing default without introducing an oracle-derived horizon.

For the same sufficient-horizon run and explicit published choices
`tabulated`, `bohe_total_energy`, no PAN-v8 term, no positive-chi 5.40 term,
and final-spin fit order `(1, 2)`, the public-binary errors are:

| quantity | plus | cross |
|---|---:|---:|
| complex relative L2 | 1.6076686159371885 | 1.6058141211502779 |
| magnitude relative L2 | 0.14101376185697054 | 0.1312189530265021 |
| relative norm error | 0.051538921389793484 | 0.05316373877133163 |
| flat mismatch | 1.3611218090439354 | 1.3602207234981538 |

The amplitude improves substantially, but the complex waveform is still
non-parity and its phase convention remains unresolved. Optimizing one
constant phase on this run reduces the L2 values only to 1.0963013891271116
and 1.095899269514754; testing a global sign gives 1.1020700779022519 and
1.101979500475997. No phase, sign, time, or remnant-fit correction is applied
from this residual. The sufficient-horizon result is a diagnostic, not a parity claim; further work must resolve
the independently published mode/Fourier convention and calibrated merger
choices.
