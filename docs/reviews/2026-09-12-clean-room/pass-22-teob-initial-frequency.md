# TEOBResumS initial-frequency audit

This pass checks one general initial-data convention against the published
EOB equations. It uses the primary references already recorded in
[`pass-3-teob-potential.md`](pass-3-teob-potential.md): Damour--Nagar,
[arXiv:0704.3550v2](https://arxiv.org/pdf/0704.3550v2), and Nagar et al.,
[arXiv:1806.01772v2](https://arxiv.org/pdf/1806.01772v2). The waveform
comparison uses only the installed public `SimInspiralFD` binary interface;
no LAL/PyCBC implementation source was inspected.

## Defect and correction

The TEOBResumS driver previously read `f[0]` as its start frequency. The
parity grid begins at DC, so this selected zero and then replaced it with an
unrelated 15 Hz floor and a dimensionless-frequency clamp. That made the
circular initial data independent of the caller's requested `f_lower` and
could silently choose a different start point for any grid whose first sample
was not the physical lower cutoff.

For the dominant quadrupole, the published orbital/GW-frequency relation is

\[
  f_{\rm GW}=\Omega_{\rm orb}/\pi,\qquad M\Omega_{\rm orb}
  =\pi M_{\rm sec} f_{\rm lower}.
\]

The driver now accepts and dispatches `f_lower`, uses that equation directly,
validates that it is finite and positive, and otherwise selects the lowest
positive frequency sample. The empirical floor and clamp were removed.

## Focused verification

`tests/test_clean_room_teob_potential.py::test_teob_initial_data_uses_requested_lower_frequency`
monkeypatches only the independent initial-data solver and captures its
frequency argument. With samples `[0, 30, 40]` Hz and `f_lower=35` Hz, it
checks exactly `pi * (2.8 * MSUN_SEC) * 35`, proving the value is threaded
through the public waveform dispatcher rather than inferred from the grid.

Command:

```text
python -m pytest -q tests/test_clean_room_teob_potential.py tests/test_teobresums.py
20 passed in 1.76s
```

## Frozen public-binary comparison

The case is the strict `TEOBResumS_diagnostic` fixture: equal 1.4 solar-mass
components, zero spins and tides, 100 Mpc, 30--500 Hz comparison band, and
0.5 Hz spacing. Metrics are direct complex relative L2, magnitude relative
L2, relative norm error, and flat mismatch. The before values are the frozen
baseline run with the previous start-frequency logic; after values are from
`/tmp/measure_teob.py` after this correction.

| polarization | complex L2 before | complex L2 after | magnitude L2 before | magnitude L2 after | norm error before | norm error after | mismatch before | mismatch after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plus | 1.0008916061 | 1.0009161561 | 0.9783049671 | 0.9875754573 | 0.9703082779 | 0.9808518173 | 1.0151963025 | 1.0382934173 |
| cross | 0.9992281891 | 0.9993295254 | 0.9780088452 | 0.9872784412 | 0.9705016574 | 0.9809765277 | 0.9590963749 | 0.9552554827 |

The correction is retained because it fixes a general published initial-data
mapping, even though this prototype's strict binary comparison is not
improved uniformly. The remaining discrepancy is expected: the module still
documents its tidal amplification, spin/self-spin terms, radiation,
merger-frequency heuristic, and taper as prototype pieces rather than a
complete calibrated TEOBResumS reconstruction. No parity claim is made.
