# Parameters and conventions

Use these conventions when setting up a waveform or comparing providers.
Model-specific options and supported domains are defined by the
[waveform catalog](models.md), the [T-family guide](imrphenomt.md) and the
[API reference](api/diffgw.md). Defaults should be matched explicitly in
cross-package comparisons.

## Physical parameters

| Quantity | Convention used in the documented CBC examples | Checks when changing models or providers |
| --- | --- | --- |
| Component masses | `mass1`, `mass2` in solar masses, with `mass1 >= mass2` | Keep each spin and tidal parameter attached to its body if reordering masses. |
| Distance | `distance` in Mpc | Match the distance and polarization projection before comparing amplitudes. |
| Spins | Dimensionless aligned `chi1`, `chi2`, or Cartesian `spin1x` through `spin2z` for supported precessing models | Match the reference frequency, source frame and precession prescription. A matching vector alone does not establish matching conventions. |
| Tidal deformabilities | Dimensionless `lambda1`, `lambda2` | Use a model that supports the required tidal prescription. |
| Frequencies | Hz; uniform `(f_lower, f_final, delta_f)` or `sample_frequencies` for supported FD paths | Match the actual bins, endpoint and cutoff policies. |
| Time | Seconds for sampled TD waveforms | Match the time origin, duration and sampling. Native T-family sampling differs from LAL's automatic epoch and length. |
| Angles | Radians | Match inclination and distinguish orbital reference phase from any waveform phase alignment used in validation. |

See [generation examples](guides/waveform-generation.md) for supported arguments.
For eccentric models, record the eccentricity and its reference-frequency
convention when comparing providers; the catalog alone does not establish a
shared eccentricity convention.

## Reference phase, frames and modes

Phase and reference-frequency behavior must be checked per model. For the
[T family](imrphenomt.md#generate-a-sampled-waveform), `phic` is orbital phase at
`f_ref`; an omitted or zero `f_ref` uses `f_lower` in the sampling wrapper.
Its peak is located at `tc` on the chosen time grid. These rules must not be
assumed for every other approximant.

A mode label also needs a frame and projection convention. The T-family guide
specifies aligned or coprecessing mode pairs and how selecting one harmonic
includes both signs. The [generation guide](guides/waveform-generation.md)
shows FD mode selection. Use the model's supported mode set rather than assuming
all models with “HM” in their names contain the same harmonics.

For LAL comparisons, consult [known differences](differences_from_lal.md)
for reference-phase, carrier, precession and calibration choices. Record the
full options alongside the approximant name.

## Outputs and batching

Waveform generation returns plus and cross polarizations. In the documented
FD examples, a single waveform has shape `(N,)` and a batch has shape `(B, N)`.
Native T-family batches use a shared time grid; `return_times=True` also returns
that grid. See [generation](guides/waveform-generation.md) and
[execution](guides/execution.md) for examples and transformation restrictions.

## Domains and invalid inputs

A calibration domain, an implemented input domain and a tested parameter set
are different things. The [validation records](validation.md) establish only
the configurations they describe. Missing domain-wide evidence should be
treated as unestablished.

Invalid-input behavior is also model-specific. For example, unsupported
T-family MSA points raise errors in eager evaluation and produce NaNs for
affected rows under JAX tracing. Its singular equal-mass precessing MSA limit
is unsupported. The [T-family restrictions](imrphenomt.md#higher-modes-and-precession)
explain alternative prescriptions and other rejected options. Do not assume
that every model or transformed execution path rejects inputs in the same way.
