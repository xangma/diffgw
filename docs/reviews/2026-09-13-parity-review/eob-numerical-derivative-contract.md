# Independent numerical derivative experiment

This is a mathematical and interface handoff from the authorized reference
auditor. It supplies no LAL or GSL implementation, source excerpts, or
calibration values. The experiment must remain separate from production
defaults. Its purpose is to measure a numerical difference, not to replace
the existing analytic Hamiltonian with fitted corrections.

The aligned reference driver obtains Hamiltonian partials by the public
`gsl_deriv_central` interface, with initial trial spacing `1e-4`. Its scalar
function is the full real Hamiltonian divided by the symmetric mass ratio;
it includes the constant rest energy. Subtracting that constant is
mathematically harmless but changes finite precision cancellation. The
native clean driver instead differentiates the effective Hamiltonian with
autograd and the analytic energy map.

Use the [public numerical differentiation manual](https://www.gnu.org/software/gsl/doc/html/diff.html),
not GSL source. It describes a central quartic stencil using offsets
`-h, -h/2, h/2, h`, a quadratic central stencil for truncation estimation,
and an adjusted spacing balancing truncation and rounding error. Derive
weights and error propagation independently. For example, writing `D(h)`
for the symmetric two-point slope, the quartic derivative is
`(4*D(h/2)-D(h))/3`; its discrepancy from `D(h)` estimates the quadratic
truncation term. A model `a/h + b*h^2` has its minimum at
`h=(a/(2*b))^(1/3)`. Document any further numerical decisions explicitly;
do not infer undocumented constants by fitting GSL results.

## Coordinate meaning

At an aligned equatorial state `(r, phi, p_r*, j)`, orient the instantaneous
Cartesian axes so the position is `(r,0,0)` and the momentum is
`(p_r*,j/r,0)`. Axial symmetry lets the three required Cartesian partials
be evaluated with the existing scalar clean Hamiltonian, as follows:

- Vary the radial position `x`, keeping the Cartesian momenta fixed:
  evaluate `H(x, p_r*, x*(j/r))/nu`.
- Vary `p_x`: evaluate `H(r, p_x, j)/nu`.
- Vary `p_y`: evaluate `H(r, p_r*, r*p_y)/nu`.

Let these derivatives be `H_x`, `H_px`, `H_py`, and let `xi` be the clean
tortoise factor. Coordinate transformation gives
`dr/dt=xi*H_px`, `Omega=H_py/r`, and
`dp_r*/dt=xi*(-H_x+H_py*j/r^2)` before radiation reaction.
The existing flux supplies the same azimuthal and radial forces afterward.
Do not introduce a different flux, calibration, initial state or attachment
in this derivative attribution experiment.

## Bounded validation

Add a separate Torch implementation with explicit domain failures,
per-row parameter handling, and gradients where the discrete stencil
selection permits them. Test polynomial exactness, smooth analytic
functions, and the chain-rule equivalence of these Cartesian expressions
to existing analytic conservative derivatives. Include nonzero spins and
nonzero radial momentum. Keep actual discrepancies visible.

The coordinator's public GSL binary evaluation on the *clean* Hamiltonian
at six points of the 30+20 zero-spin public trajectory differs from clean
analytic conservative derivatives by roughly `1e-12` to `1e-11` in absolute
units. See
[the diagnostic](verification/eob-adaptive-root-review/public-derivative-on-clean-hamiltonian.json).
These are test outputs, not coefficients or correction terms. They do not
yet establish the impact on complete waveforms. The public binary may be
used in optional validation scripts only; runtime must not import it.

Do not launch a complete inspiral for this first task. Report the independent
method, pointwise errors, tests, and cost so the coordinator can decide the
next controlled waveform comparison.
