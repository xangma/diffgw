# Direct PA quadratic value contract

`initializer_policy="direct-quadratic-value"` is an opt in, value only
diagnostic. The default `"analytic"` initializer and the existing stabilized
`"quadratic-value"` diagnostic retain their behavior.

The circular stationarity equation follows the public EOB Hamiltonian
relations in [Nagar et al., arXiv:1806.01772, Appendix C](https://arxiv.org/abs/1806.01772):

\[
H_{\rm eff}=\sqrt{A+Uj^2}+Gj,\qquad U=A/r_c^2,\qquad x=j^2.
\]

Writing primes for radial derivatives and squaring the stationarity equation
gives `a*x**2 + b*x + c = 0`, with

```
a = Uprime**2 - 4*U*Gprime**2
b = 2*Aprime*Uprime - 4*A*Gprime**2
c = Aprime**2.
```

The direct diagnostic evaluates the numerical intermediate and polynomial in
the requested Torch dtype:

```
u = 1/r_c
U = A*u**2
Uprime = u**2*(Aprime - 2*A*u*r_c_prime)
discriminant = b**2 - 4*a*c
```

A negative computed discriminant is clamped to zero. The minus square root
branch is selected for `Gprime <= 0`; the plus branch is selected for positive
`Gprime`. The helper checks finite values, requires positive `A`, `r_c`, and
roots, and allows signed derivatives and `Gprime`; it rejects a zero
quadratic coefficient. The PA initializer runs this probe under
`no_grad` and rejects tracked intrinsic parameters because the branch and
discriminant crossing do not provide an autograd contract.

The equations are used as a clean numerical contract. This policy does not
claim a new physical PA prescription or alter the canonical route. PA context
is described publicly in [Nagar & Rettegno, arXiv:1805.03891](https://arxiv.org/abs/1805.03891).
