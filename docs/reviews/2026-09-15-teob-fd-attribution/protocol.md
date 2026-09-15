# Continuous TEOB frequency-domain attribution

The first experiment uses the exact current TEOB catalog case: nonspinning,
zero-tide 1.4+1.4 solar masses, 100 Mpc, face-on, zero phase/time offsets,
30–500 Hz with 0.5 Hz bins. The standard parity harness retains its full
polarization, detector, PSD, and out-of-band checks and unchanged thresholds.
The clean adapter is selected only by a diagnostic wrapper. Public dispatch
is not changed. Public waveform samples are diagnostic references only.

The initial native policy is continuous matching, analytic 8PA initialization,
leading-order horizon flux, and frozen-mobility radial acceleration. Cache all
stages and source hashes. The candidate's full Nyquist grid is mapped by exact
indices to the oracle grid; the API wrapper zeros bins outside [flow,fhigh).

After the first comparison, independently control epoch, tail support,
conditioning, and pre-/post-attachment contributions. An oracle-derived time,
phase, or amplitude fit may diagnose a residual but cannot enter runtime or
count as a parity improvement. Any runtime fix requires an independently
specified numerical or physical reason and tests of that reason.

Before examining new outcomes, reserve these additional nonspinning, zero-tide
cases for validation (all 100 Mpc, face-on, zero phase/time, delta_f=0.5 Hz):

| Case | Masses (solar masses) | Requested band (Hz) |
|---|---|---|
| Existing reference configuration | 30+20 | 20–1024 |
| Equal BBH | 20+20 | 25–512 |
| Unequal BBH | 40+10 | 20–1024 |
| Unequal BNS | 1.6+1.2 | 30–500 |

Previously inspected configurations are regressions. This set is a bounded
validation experiment, not certification of the full physical parameter domain.
