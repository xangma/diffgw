# Continuous NQC matching derivative qualification

This is a development-only diagnostic of the frozen `tabulated` SEOBNRv4
generator at mass pair `(30,20)`, start frequency `0.04`, and spins
`(0,0)` and `(0.2,-0.1)`. The frozen baseline, ODE/grid gate definitions,
limits, and accepted trajectory hashes are in
[`../ode_gate_slice/README.md`](../ode_gate_slice/README.md). The baseline
commit is `a28440a77547bee894b51192a6e0253ba5f3b859`.

## Rule and independent check

The predeclared numerical rule and acceptance bounds are in [PLAN.md](PLAN.md).
The matching equations follow [Bohé et al. (2017), Eq. (7)](https://arxiv.org/html/1611.03703).
For the published ODE `x'=f(x)` and matching inputs
`g=(A,Aq1,Aq2,Aq3,p1,p2,-arg(h))`, compute `Dg=∇g·f` and
`D²g=fᵀ Hess(g) f+∇g·(J_f f)` by automatic differentiation at the continuous
match state. The state is located on the accepted native trajectory by its
existing natural cubic state interpolant. The sampled spline supplies the
value row; the chain rule supplies every first and second derivative row.
Attachment, calibration targets, match time, ODE tolerances, grids, and gate
limits remain frozen.

[`diagnose.py`](diagnose.py) hash-replays the frozen refined ODE waveform for
each row. It then integrates the published vector field independently with
SciPy DOP853 at two tight tolerances. It evaluates the matching vector on the
dense solution at three symmetric fourth-order stencil widths and compares
the finest stencil with automatic differentiation at the DOP853 center. The
independent integration checks numerical differentiation of the same
published equations; it does not validate the physics against another model.
The local phase value may differ by an integer multiple of `2π`; the phase
derivatives are chart invariant here.

| Check | Declared bound | Nonspinning | Spinning |
| --- | ---: | ---: | ---: |
| Tight DOP853 center state spread, maximum absolute | `1e-8` | `1.07150e-11` | `5.24807e-11` |
| Native versus tight DOP853 chain vector, maximum relative | `1e-5` | `4.24340e-9` | `4.71434e-9` |
| Tight DOP853 `0.025 M` stencil versus chain vector, maximum relative | `1e-4` | `2.21541e-8` | `4.06190e-8` |

The native versus DOP853 match-state difference is `6.12989e-10` and
`6.53337e-10` maximum absolute, respectively. The amplitude matrix condition
numbers are `8065.65` and `10578.08` for the sampled spline and `8065.68`
and `10577.97` for the chain candidate; the phase matrix condition numbers
are `78.44`, `97.41`, `78.44`, and `97.40` in the same order. All solves are
finite. Full matching vectors, systems, coefficients, dense-solution checks,
and replay hashes are in [nonspin.json](nonspin.json) and [spin.json](spin.json).

The refined ODE matching derivatives below follow the `g` order stated above.
Values are rounded here; the JSON files retain float64 values, value rows,
linear systems, and coefficient vectors.

| Row | Source | `Dg` |
| --- | --- | --- |
| Nonspin | Sampled spline | `(.0058300193063,.015172015882,.006299028738,.0040130231983,-.037040460874,-.0038193527848,.2855328436)` |
| Nonspin | ODE chain | `(.0058300193505,.015172015373,.0062990285634,.0040130230964,-.037040459469,-.0038193526394,.28553284382)` |
| Spin | Sampled spline | `(.0055271734129,.010843409588,.0043327580002,.0027111896333,-.030634610818,-.0025871631093,.28291396307)` |
| Spin | ODE chain | `(.0055271734845,.010843408849,.0043327577483,.0027111894863,-.030634608512,-.0025871629085,.28291396343)` |

| Row | Source | `D²g` |
| --- | --- | --- |
| Nonspin | Sampled spline | `(-.00083383356514,.0044627480268,.0020519930806,.0013730162328,-.0081184016474,-.0013375576511,.0068342673033)` |
| Nonspin | ODE chain | `(-.00083382703942,.0044626988995,.0020519748608,.0013730050852,-.008118277645,-.0013375427034,.0068342820592)` |
| Spin | Sampled spline | `(-.00056243561276,.0030023077227,.0013194795522,.00086448613,-.0062656919619,-.00085926966392,.0062974242202)` |
| Spin | ODE chain | `(-.00056243873836,.0030023045902,.0013194800748,.00086448699984,-.0062656652815,-.00085926999094,.006297433505)` |

## Frozen gate replay

The candidate is an opt-in diagnostic in the SEOBNR development driver.
[`gate_candidate.py`](gate_candidate.py) first verifies the refined ODE
coefficient replay, then runs the unchanged ODE and grid levels for both
rows. Its receipts retain the baseline and candidate NQC coefficients,
the original and candidate waveform hashes, and the candidate accepted-array
hashes for every level.

The frozen complex and amplitude relative L2 limit is `1e-6` at the refined
pair. Complex error must decrease through both pairs (with the original
`1e-12` roundoff allowance). Peak, match, and boundary differences must each
remain within `0.05 M`. Each cell below gives coarse→medium then medium→fine.

| Row | Pair | Baseline complex | Candidate complex | Baseline amplitude | Candidate amplitude |
| --- | --- | --- | --- | --- | --- |
| Nonspin | ODE | `2.13598837e-5 → 9.07920587e-6` | `5.10013902e-6 → 2.61447123e-6` | `8.17301127e-7 → 3.47176949e-7` | `1.36759957e-7 → 6.86663191e-8` |
| Nonspin | Grid | `1.33339639e-7 → 6.14985548e-6` | `6.22068128e-9 → 8.17864034e-9` | `5.71463295e-9 → 2.61431347e-7` | `5.15098074e-11 → 3.61087505e-11` |
| Spin | ODE | `6.31409093e-5 → 4.93844880e-5` | `1.70989065e-7 → 1.67095665e-8` | `1.89161908e-6 → 1.47015986e-6` | `6.71313911e-10 → 1.35749982e-10` |
| Spin | Grid | `5.76186573e-7 → 5.50714987e-7` | `5.93948684e-9 → 3.63418691e-9` | `1.68225132e-8 → 1.60978106e-8` | `4.81704095e-11 → 2.98452462e-11` |

Every original gate, including the event gates, is shown below. `Decreasing`
tests the complex error; `Peak`, `Match`, and `Boundary` test event times.

| Row | Pair | Source | Complex | Amplitude | Decreasing | Peak | Match | Boundary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nonspin | ODE | Baseline | **FAIL** | pass | pass | pass | pass | pass |
| Nonspin | ODE | Candidate | **FAIL** | pass | pass | pass | pass | pass |
| Nonspin | Grid | Baseline | **FAIL** | pass | **FAIL** | pass | pass | pass |
| Nonspin | Grid | Candidate | pass | pass | **FAIL** | pass | pass | pass |
| Spin | ODE | Baseline | **FAIL** | **FAIL** | pass | pass | pass | pass |
| Spin | ODE | Candidate | pass | pass | pass | pass | pass | pass |
| Spin | Grid | Baseline | pass | pass | pass | pass | pass | pass |
| Spin | Grid | Candidate | pass | pass | pass | pass | pass | pass |

The candidate passes both gates on the spinning row. The nonspinning ODE
complex error remains `2.61447e-6`, above `1e-6`. Its grid errors are far
below the budget but increase from `6.22068e-9` to `8.17864e-9`, beyond the
frozen monotonicity allowance. All ten candidate time arrays exactly match
their baseline arrays, and the match/peak/boundary events are unchanged.
The full before/after metrics and hashes are in [gate_nonspin.json](gate_nonspin.json)
and [gate_spin.json](gate_spin.json).

The NQC coefficient vectors below are ordered `(a1,a2,a3)` and `(b1,b2)`.
The baseline values are the provisional sampled-spline solve from the same
run; the candidate values are the attached waveform's chain-rule solve.

| Row | Level | Source | Amplitude | Phase |
| --- | --- | --- | --- | --- |
| Nonspin | `ode_0` | Baseline | `4.1236952632,-47.136064136,56.943754143` | `4.1368987278,-21.977977020` |
| Nonspin | `ode_0` | Candidate | `4.1232444979,-47.132170781,56.939350080` | `4.1367972561,-21.977012538` |
| Nonspin | `ode_1` | Baseline | `4.1231296227,-47.131223215,56.938303697` | `4.1367570055,-21.976705363` |
| Nonspin | `ode_1` | Candidate | `4.1231734285,-47.131609108,56.938744507` | `4.1367676539,-21.976813393` |
| Nonspin | `ode_2` | Baseline | `4.1233743980,-47.133327594,56.940678531` | `4.1368172447,-21.977250976` |
| Nonspin | `ode_2` | Candidate | `4.1232089945,-47.131890145,56.939047492` | `4.1367824507,-21.976912911` |
| Nonspin | `grid_1` | Baseline | `4.1233701877,-47.133291041,56.940637077` | `4.1368163192,-21.977242020` |
| Nonspin | `grid_1` | Candidate | `4.1232089910,-47.131890118,56.939047463` | `4.1367824493,-21.976912901` |
| Nonspin | `grid_2` | Baseline | `4.1231778024,-47.131620756,56.938742779` | `4.1367736406,-21.976828861` |
| Nonspin | `grid_2` | Candidate | `4.1232089953,-47.131890152,56.939047499` | `4.1367824511,-21.976912913` |
| Spin | `ode_0` | Baseline | `8.7962197724,-87.292001650,102.282384000` | `5.5653261413,-35.033199961` |
| Spin | `ode_0` | Candidate | `8.7956640382,-87.287013476,102.276631943` | `5.5651801484,-35.031437274` |
| Spin | `ode_1` | Baseline | `8.7943267600,-87.275064691,102.262884494` | `5.5648342082,-35.027335176` |
| Spin | `ode_1` | Candidate | `8.7956642821,-87.287015373,102.276633986` | `5.5651802204,-35.031437786` |
| Spin | `ode_2` | Baseline | `8.7957893664,-87.288134224,102.277921953` | `5.5652182574,-35.031890431` |
| Spin | `ode_2` | Candidate | `8.7956641684,-87.287014445,102.276632970` | `5.5651801792,-35.031437443` |
| Spin | `grid_1` | Baseline | `8.7958060754,-87.288283464,102.278093626` | `5.5652227365,-35.031943463` |
| Spin | `grid_1` | Candidate | `8.7956641720,-87.287014474,102.276633002` | `5.5651801805,-35.031437454` |
| Spin | `grid_2` | Baseline | `8.7958220416,-87.288426085,102.278257694` | `5.5652270109,-35.031994096` |
| Spin | `grid_2` | Candidate | `8.7956641531,-87.287014321,102.276632833` | `5.5651801737,-35.031437397` |

## Execution and outcome

The independent diagnostic used frozen baseline source manifest
`ba2c668401077505afd7b095eff1b81e2f13d03aba5bcf4a6e16df52383b8427`.
The candidate gate source manifest was
`2caa5b4a3bb20e01cd6e0b0e8891e76b5d49345e3a254c9cc2395d7304d06a13`;
each gate receipt confirms that its source stayed unchanged while running.
The saved plan SHA-256 is
`0843da44b4fd657121ee84cf21c9f8da651c5896fbd63b168b2e9f66d76b37b9`.
Both independent diagnostics replayed the exact frozen refined ODE waveform
hashes before comparing matching derivatives. The gate replay verified its
provisional and candidate refined ODE coefficients against those diagnostic
solves before running the remaining levels.

All runs used host `len`, cwd
`/home/xangma/tmp/diffgw-seobnr-ode-gate-aa1f365f`, Python
`/home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python`, `PYTHONPATH=src:.`,
one OpenMP/OpenBLAS thread, CPU JAX x64, and the command shown below after
the Python executable. The expected next check was the corresponding JSON
checkpoint and log. All four PIDs have exited; the stop command while any
was running was `kill <PID>` on `len`.

| Run | Command | PID | Local log |
| --- | --- | ---: | --- |
| Independent nonspin | `dev/native_rom/evidence/2026-09-29/ode_chain_derivatives/diagnose.py --row nonspin --json-out dev/native_rom/evidence/2026-09-29/ode_chain_derivatives/nonspin.json` | 4063563 | [nonspin.log](nonspin.log) |
| Independent spin | same diagnostic command with `--row spin` and `spin.json` | 4066905 | [spin.log](spin.log) |
| Gate nonspin | `dev/native_rom/evidence/2026-09-29/ode_chain_derivatives/gate_candidate.py --row nonspin --json-out dev/native_rom/evidence/2026-09-29/ode_chain_derivatives/gate_nonspin.json` | 129837 | [gate_nonspin.log](gate_nonspin.log) |
| Gate spin | same gate command with `--row spin` and `gate_spin.json` | 249106 | [gate_spin.log](gate_spin.log) |

The first independent diagnostic execution computed the comparisons but its
JSON writer rejected a missing `scope` field. The field was added and both
rows were rerun with the same diagnostic method; their final receipts and
logs are linked above. No numerical gate or tolerance was adjusted.

The published chain rule candidate is numerically supported by the independent
dense trajectory check and fixes the spinning row's frozen ODE gate. It does
not qualify the two-row generator because nonspin still fails its ODE complex
budget and grid monotonicity gates. Therefore the candidate remains behind
its opt-in development flag; the default and production generator have not
been changed. No deployment, ROM promotion, PR, or external source comparison
was performed. The next investigation needs a source-backed account of the
remaining nonspin ODE discrepancy and output-grid monotonicity behavior,
without relaxing the frozen checks.
