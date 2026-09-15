# Clean XHM intermediate amplitude

The clean helper represents the dimensionless inverse normalized amplitude
`P(u) = A0/A(f)` (with `A0` held constant across the join) on
`u=(f-f_start)/(f_end-f_start)` as
`P(u)=sum(k=0..5) c_k u^k`. Six linear constraints determine the six
coefficients: value and first derivative at each endpoint, plus two caller
supplied interior values. Endpoint derivatives are with respect to `u`.
The public wrapper accepts normalized amplitudes `v=A/A0` and derivatives
`dv/df`, converting them with `P=1/v` and
`dP/du=-(f_end-f_start)(dv/df)/v^2`. The evaluator returns `A0/P(u)` and
rejects non-positive or near-zero denominators. The construction is the
inverse-quintic described in [arXiv:2001.10914v2, Sec. V.1.1, Eq. (37) and
Table 4](https://arxiv.org/abs/2001.10914v2); this generic caller-supplied
schema is not asserted to be the unreleased `122022` provider format. No
calibration values, release vetoes, or provider assumptions are included.
