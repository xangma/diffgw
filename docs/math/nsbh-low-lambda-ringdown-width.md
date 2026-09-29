# IMRPhenomNSBH low-tidal ringdown width

## Scope and status

Specification **nsbh.amplitude.low-lambda-width/v1** freezes the
$0\leq\Lambda\leq1$ branch of [Thompson et al. (2020), Eq. (29), with
Eq. (28)](https://arxiv.org/html/2002.08383v2). It is a conditional
component gate with supplied PhenomC $\delta_2$, final-state ringdown
frequency $f_{\rm RD}$, and quality factor $Q$. It does not qualify the
$\Lambda>1$ branch, the Eq. (26) premerger correction, the Lorentzian
carrier, or the complete NSBH waveform.

## Inputs, conventions, and options

$\Lambda$ is dimensionless and belongs to the closed interval $[0,1]$;
the equality case selects this branch. The supplied $\delta_2\geq10^{-4}$
is dimensionless, consistent with the paper's stated positivity guard.
The supplied $f_{\rm RD}>0$ is in Hz and $Q>0$ is dimensionless. This
gate does not evaluate their upstream fits or alter production input guards.

## Mathematics and components

The paper prints $b_0=0.81248$ and

\[
\delta'_2=\delta_2-2(\delta_2-b_0)\Lambda
              +(\delta_2-b_0)\Lambda^2,
\qquad
\sigma=\delta'_2\frac{f_{\rm RD}}{Q}.
\]

The equivalent form $\delta'_2=b_0+(\delta_2-b_0)(1-\Lambda)^2$
establishes $\delta'_2(0)=\delta_2$, $\delta'_2(1)=b_0$, positivity,
and zero first $\Lambda$ derivative at the branch endpoint. For
independently supplied inputs, the exact partials are

\[
\partial_\Lambda\delta'_2=2(\delta_2-b_0)(\Lambda-1),\quad
\partial_{\delta_2}\delta'_2=(1-\Lambda)^2,\quad
\partial_\Lambda\sigma=\frac{f_{\rm RD}}Q
\partial_\Lambda\delta'_2,\quad
\partial_{\delta_2}\sigma=\frac{f_{\rm RD}}Q(1-\Lambda)^2,\quad
\partial_{f_{\rm RD}}\sigma=\frac{\delta'_2}Q,\quad
\partial_Q\sigma=-\frac{\delta'_2f_{\rm RD}}{Q^2}.
\]

These derivatives are conditional partials, not derivatives through the
physical remnant or PhenomC providers.

## Implementation mapping

`diffgw_reference.nsbh_low_lambda_width` will evaluate both printed
expressions independently in high-precision scalar arithmetic. A
production-called `delta2_prime_small_lambda` helper in each native
backend will replace only the inline low-$\Lambda$ polynomial; the
existing Eq. (28) width multiplication and high-$\Lambda$ selection
remain unchanged. Tests will compose the helper with Eq. (28) and compare
values and automatic derivatives with the scalar reference. No LAL source
or binary-derived coefficients enter the reference.

## Evidence and limitations

The grid and limits are frozen before numerical native comparison:
$\delta_2\in\{10^{-4},0.5,0.81248,2\}$,
$\Lambda\in\{0,2^{-20},1/8,1/2,7/8,1-2^{-20},1\}$,
$f_{\rm RD}\in\{100,600\}$ Hz, and $Q\in\{2,8\}$. This includes the
BBH limit, both sides of the low-branch interior, and the closed endpoint;
it deliberately excludes $\Lambda>1$. Independent 70- and 90-digit
scalar values and partials must differ by at most $10^{-30}$ absolutely.
Torch/JAX CPU FP64 values must agree with the 70-digit scalar to
$2\times10^{-13}$ absolute for $\delta'_2$ and $\sigma$, and
$2\times10^{-12}$ absolute for the six partials listed above. Check
batched shape and the $\Lambda=0,1$ identities. These limits must not be
retuned after comparison.

[The variant audit](nsbh-amplitude-variant-provenance.md) records that
the printed $\Lambda>1$ Eq. (29) window sign differs from the native
compatibility variant; this gate makes no claim about that branch. The
Eq. (26) polynomial/baseline ambiguity, low/no-torus joining center,
upstream remnant extrapolation, full amplitude, and whole waveform remain
separate qualification gates.

## Reproduce the checks

On len CPU with FP64 enabled, run `JAX_ENABLE_X64=1 PYTHONPATH=src
python -m pytest -q tests/test_nsbh_low_lambda_width_math_reference.py`.
Record package versions, numerical maxima, neighboring tests, and any
environment limitation in a validation receipt before claiming a pass.
