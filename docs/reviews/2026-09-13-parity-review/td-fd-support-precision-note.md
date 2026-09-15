# TD support precision note

The clean support planner uses the Newtonian TaylorT3 inversion derived from
the leading TaylorT2 time relation

\[
t(v)=\frac{5M}{256\eta v^8},\qquad f=\frac{v^3}{\pi M}.
\]

Writing \(\Theta=\eta t/(5M)\), and then forming its inverse, gives the
published TaylorT3 frequency
\[
f=\frac{(1/\Theta)^{3/8}}{8\pi M}.
\]
The implementation evaluates this as one direct `pow(..., 3/8)` operation;
it does not form an eighth-root velocity and then cube it.

For binary64 inputs, the SI mass boundary is explicit:

```
m1SI = m1_solar * MSUN_SI
m2SI = m2_solar * MSUN_SI
totalSI = m1SI + m2SI
reducedSI = m1SI * m2SI / totalSI
eta = reducedSI / totalSI
```

The inverse dimensionless time uses `5 * totalSI * (MSUN_SEC/MSUN_SI)` in its
numerator.  The dimensionful frequency prefactor converts `totalSI` separately
with `G_SI/c_SI**3`.  These are the same physical conversion, but retaining the
two unit boundaries avoids hiding a binary64 rounding choice.  Float32 uses the
algebraically equivalent scaled mass-ratio path so the intermediate SI product
cannot overflow; all returned tensors retain the requested dtype, device,
batching, and autograd behavior.

The public timing APIs are validation controls, not runtime dependencies.  For
the cases below, `SimInspiralChirpTimeBound` is called at the planner's safe
frequency, then the public merge, ringdown, and start-frequency bounds are
called in sequence.  The reported ULP delta is clean start minus public start.
The exact invocation and all intermediate values are reproduced by
[`diagnose_td_fd_support_precision_clean.py`](../../../scripts/diagnose_td_fd_support_precision_clean.py)
and its [JSON report](verification/td-fd-support-precision.json).

| masses | spins | requested Hz | clean − public (ULP) |
| --- | --- | ---: | ---: |
| 30 + 20 | 0, 0 | 20 | 0 |
| 1.4 + 1.4 | 0, 0 | 30 | +3 |
| 80 + 60 | 0, 0 | 80 | +2 |
| 42 + 17 | -0.3, 0.2 | 35 | 0 |
| 5 + 4 | 0.7, 0.2 | 20 | 0 |
| 100 + 5 | -0.95, 0 | 10 | -1 |

The remaining one-to-three-ULP differences are unresolved after the declared
arithmetic boundary and are left visible rather than corrected with a
`nextafter`, fitted offset, or relaxed threshold.

The TaylorT3 variable convention is documented in the primary PN literature,
for example [Damour, Iyer, and Sathyaprakash, *Phys. Rev. D* 57, 885
(1998)](https://arxiv.org/abs/gr-qc/9708034), with the same leading-order
dimensionless time variable and frequency prefactor.
