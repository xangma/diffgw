# Clean adaptive SEOB full20 attribution

This directory records one fresh clean-runtime reconstruction of the
30+20-solar-mass, zero-spin, 20 Hz SEOB case at `dt_geom=0.5M`.  It uses the
reviewed `gsl_state_scaled` RKF45 controller with
`atol=1e-10`, `rtol=1e-9`, a 5,000 accepted-step cap, a 100,000 evaluation
cap, a 10,000M duration cap, and 150M ringdown.  The run was diagnostic-only:
no alignment, fit, gate, baseline, or canonical route was changed.

The command was run on `RONI2` in `/Users/xangma/repos/torchwave` as tracked
job `36162` and completed in 30.90 seconds.  `startup.json` records the
launch contract and pre-run hashes.  The foreground session output was
captured in the task log; `diagnostic.json` is the machine-readable final
report.  The cache is
`native-full20-controller-clean.npz`.

The helper returns a geometric mode.  The frozen public `mode22` is the same
unprojected mode with the standard 100 Mpc factor, so the comparison uses the
independently specified clean conversion
`M_sec*C_SI/(100 Mpc)` and stores both fields in the cache.  No amplitude
normalization was fitted.

Against the frozen public mode, the full zero-padded raw complex relative L2
is `2.766799454794218e-5` (absolute L2 `8.030917271996032e-24`), and the
amplitude relative L2 is `8.587714186520147e-7` (absolute L2
`2.49267152586668e-25`).  Both arrays contain 10,230 exact nonzero samples
and end at `5114.5M`; the public array has 10,773 storage samples and the
native array 10,297.

The coarse stage accepted 284 steps, rejected 44 trials, and made 1,925 RHS
evaluations before its event at `5002.950544626954M`.  The fine stage accepted
74, rejected 22, and made 555 evaluations before its event at
`150.82135968186793M`.  The splice restarts at `4853M` with refinement 32;
the peak is `5002.7187423706055M` and the NQC match is
`4998.346362836525M`.

Against the old native frozen cache, times are bitwise equal; coarse and fine
output-state relative L2 values are `7.60316408474666e-15` and
`6.842999914610774e-13`, respectively.  The geometric-mode relative L2 is
`2.176630040655158e-8`.  The report includes these direct equal-shape checks,
all counters, support data, and runtime/input hashes.

