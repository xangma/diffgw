# IMRPhenomXHM intermediate amplitude from supplied targets

## Scope and status

Component `xhm.amplitude.intermediate.inverse-polynomial.v1` implements the
six-constraint algebra in [García-Quirós et al., arXiv:2001.10914v2,
Sec. V.1.1, Eq. (37)](https://arxiv.org/html/2001.10914v2). It accepts
calibrated amplitudes and endpoint derivatives as inputs; it does not supply
the fit surfaces, interval boundaries, veto policy, or a release-122022
provider. This component is not connected to the production XHM waveform.

## Inputs, conventions, and options

For positive frequency \(f\in[f_s,f_e]\), set \(v(f)=A(f)/A_0\), where the
same constant \(A_0\) applies across the join. The paper's ansatz is

\[
v(f)=\frac{1}{\sum_{k=0}^{5}\delta_k f^k}.
\]

## Mathematics and components

The six constraints are \(v(f_s)\), \(dv/df(f_s)\), two interior values at
\(f_s+(f_e-f_s)/3\) and \(f_s+2(f_e-f_s)/3\), \(v(f_e)\), and
\(dv/df(f_e)\). The native kernels solve in \(u=(f-f_s)/(f_e-f_s)\) to
avoid poor conditioning at small physical frequencies. Their endpoint
inverse-polynomial derivative is
\(d(1/v)/du=-(f_e-f_s)(dv/df)/v^2\). The independent 70-digit mpmath
reference solves directly for the paper's physical-frequency coefficients.

## Implementation mapping

The Torch helper checks finite positive inputs, ordered endpoints,
in-interval evaluation, and conservatively certifies a positive denominator
over the whole interval by Bernstein subdivision. The JAX helper is a
JIT-compatible algebra kernel for caller-validated inputs; it does not
certify denominator positivity. Both preserve gradients through the solve.

## Evidence and limitations

Two supplied-target cases with different interval widths and amplitude
slopes are checked at eleven frequencies and nine-point batched grids.
The gate compares scalar endpoint and interior collocations, both endpoint
first derivatives, full native values, frequency derivatives, and
sensitivity to an interior amplitude target. The native value tolerance is
`2e-11` relative/absolute; frequency derivatives use `1e-8` relative and
`2e-9` absolute, and target sensitivity uses `2e-7` relative and `2e-8`
absolute. These are algebra tests, not empirical XHM accuracy claims.

The published Eq. (37) also has exceptional lower-degree and extra
pre-intermediate constructions; the present component covers the ordinary
two-interior-node degree-five branch only. Releasing a full mode additionally
requires a current-release amplitude provider, region cuts and vetoes,
the calibrated spheroidal (3,2) amplitude and mixing, then strict binary and
NR holdouts. The pinned [XHM/XPHM qualification
receipt](../waveforms/research/xhm-xphm-frozen-binary-20260929.json) remains
failed independently of this component.

## Reproduce the checks

```bash
PYTHONPATH=src:tests JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_xhm_intermediate_amplitude_math_reference.py \
  tests/test_xhm_intermediate_amplitude_clean.py
```
