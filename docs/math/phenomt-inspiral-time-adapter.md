# IMRPhenomT dominant inspiral time and sampling adapter

## Scope and status

`T-INSP-TIME-ADAPTER-v1` checks the aligned `(2,2)` carrier only, with the
frequency root between the author carrier's first collocation time and the
inspiral frequency join. It composes the already checked TaylorT3 PN terms,
six-node pseudo-PN solve, and pinned finite-grid fit inputs. The checked
outputs are seconds-to-mass-time conversion, time at frequency, inspiral
phase differences, reference-phase anchoring, and the public fixed time grid
and lower-frequency mask. It does not qualify amplitude, merger/ringdown,
higher modes, precession, physical calibration, or the full `T-SAMPLING-v0`
family contract.

## Inputs, conventions, and options

Use ordered solar masses `m1 >= m2 > 0`, aligned spins in `[-1,1]`,
`M=m1+m2`, `eta=m1*m2/M²`, and `delta=(m1-m2)/M`. The source-backed grid is
`(30,30,0,0)`, `(35,25,0.7,-0.5)`, and `(60,6,0.9,0.1)` from the pinned
author-fit fixture; inversion targets are generated at carrier
`theta=0.45,0.65,0.75`, with reference-phase checks at `0.55,0.65,0.75`.
All numerical checks run in CPU float64, including JAX x64. Times are seconds
relative to the `(2,2)` amplitude peak, and `f` is the `(2,2)` cyclic
frequency in Hz. The project adapter takes `delta_t`, `duration`, `t_pad`,
`tc`, `f_lower`, and optional `f_ref`; this gate uses finite positive values,
`f_lower` below Nyquist, and explicit duration.

## Mathematics and components

[Estellés et al., THM, Eqs. (8)–(11)](https://arxiv.org/pdf/2012.11923v1)
give `theta=[eta(t0-t/M)/5]^(-1/8)` and the orbital TaylorT3-plus-pseudo-PN
frequency. The native and pinned MIT author inspiral ansatz instead evaluates
the collocated carrier with `theta=(-eta*u/5)^(-1/8)`, where
`u=(t-tc)/(M*MSUN_SEC)`. Thus this gate uses the **author carrier** time
origin, `u=-5/(eta*theta^8)`, after the first collocation time; it does not
silently identify the paper's `t0` with this origin. The source-backed `t0`
still enters the first collocation target and the phase continuity offset.
The pinned MIT author
[`internals.py`](https://gitlab.com/imrphenom-dev/phenomxpy/-/blob/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt/internals.py)
also applies an explicit `time-t0` shift when inverting frequencies **before**
its `tEarly`; that earlier branch is outside this gate and differs from the
native inversion. No LAL algorithm or constants are used.

With dimensionless `(2,2)` angular frequency `Omega(theta)` from
`T-PN-T3-v1` and `T-PN-FREQ-COLLOC-v0`, solve the scalar equation
`Omega(theta)=2*pi*f*M*MSUN_SEC` on `0.33 < theta < 0.82` and return
`t=-5*M*MSUN_SEC/(eta*theta^8)`. `MSUN_SEC=G_SI*MSUN_SI/C_SI^3` uses the
project's CODATA/IAU constants. The scalar solver bisects in theta and does
not call a native `time_at_frequency` or duplicate its time-domain bracketing.

[Estellés et al., T/TP, Eq. (19)](https://arxiv.org/pdf/2004.08302v2)
integrates the carrier frequency. Up to a constant, the independent phase is
`Phi(theta)=Phi_T3(theta)+(10/eta)*sum_j[w_j*theta^(j+3)/(j+3)]` for `j=0..5`.
Only phase differences are checked, so the author phase-continuity constant
and amplitude sign cancel. For face-on polarization, the unit complex strain
ratio at time `t` and reference-root time `t_ref` is
`exp(i[Phi(t)-Phi(t_ref)])`; a change from `f_ref1` to `f_ref2` at the same
sample rotates it by `exp(-i[Phi(t_ref2)-Phi(t_ref1)])`. These assertions
exercise the native `f_ref` subtraction and the `2*phic` convention without
claiming that the paper prescribes the public reference-frequency API.

The **diffGW interface** grid has `n=ceil(duration/delta_t)` and
`t_k=k*delta_t-[n*delta_t-t_pad]` when centered, or `t_k=k*delta_t` otherwise.
It resolves omitted or zero `f_ref` to `f_lower`, evaluates the unmasked
waveform, then sets both polarizations to exactly zero where
`t_k < tc+t(f_lower)`. These grid, mask, and `f_ref=0` rules are project
adapter contracts, not equations claimed from the paper or author code.

## Implementation mapping

| Layer | Entry points |
| --- | --- |
| Scalar | `diffgw_reference/phenomt_inspiral_time_adapter.py`; independent mpmath theta root and phase, pure-Python grid/mask |
| Torch | `IMRPhenomT.frequency`, `time_at_frequency`, `time_domain`; `_phenomt_td.generate_td` |
| JAX | same public names and adapter in `jaxwave` |
| Frozen inputs | `tests/data/phenomt_author_inspiral_fits_4121.json`, independently checked in `T-PN-FIT-INPUT-v1` |

Neither native backend imports the scalar module or fixture.

## Evidence and limitations

Freeze before implementation: root time `rtol=2e-9, atol=2e-9 s`, forward
frequency `rtol=2e-9, atol=2e-9 Hz`, phase unit-complex ratios
`atol=3e-7`, grid coordinates `atol=2e-14 s`, and exact zeros before the
mask boundary. The strict gate has no skips. It tests three source-backed
intrinsic points but is no physical-error bound. Inversion at or above the
inspiral join, the author pre-`tEarly` branch, automatic duration, waveform
amplitude, and higher-mode adapters remain open.

## Reproduce the checks

On `len`, with current CPU Torch/JAX and mpmath installed, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests python -m pytest -q tests/test_phenomt_inspiral_time_adapter_math_reference.py --reference-device=cpu -ra`.
Record the exact Git revision, package versions, result, and source boundary
in a dated validation receipt. Revise this spec and component version before
changing the contract or thresholds.
