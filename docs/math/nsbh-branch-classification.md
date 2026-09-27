# IMRPhenomNSBH merger-branch classification

## Scope and status

Specification **nsbh-branch-classification/v1** fixes the four merger classes in
Thompson et al. (2020) Table I. It is a component specification, not a claim
about the branch-dependent amplitude fits, remnant/QNM, phase, or assembled
waveform. This revision fixes the reference inputs, boundary conventions, and
acceptance grid before native comparison.

## Inputs, conventions, and options

The reference takes the positive, finite tidal frequency $f_{\rm tide}$,
positive, finite remnant ringdown frequency $f_{\rm RD}$, and the baryonic
torus-mass ratio $T=M_{b,\rm torus}/M_{b,\rm NS}$. Both frequencies use the
same units (Hz in the native carrier). $T$ may be the signed printed fit or
the native nonnegative adapter: both have the same strictly positive torus
flag. Inputs outside these preconditions are outside this gate.

The synthetic FP64 grid fixes $f_{\rm tide}=1000$ Hz,
$f_{\rm RD}\in\{\operatorname{nextafter}(1000,-\infty),1000,
\operatorname{nextafter}(1000,+\infty)\}$ Hz, and
$T\in\{-0.25,0,+0.25\}$. Thus every class, equality, and both sides of the
frequency and torus boundaries are checked. A second composition grid uses
$q\in\{1,2,5,10,15\}$, $\chi\in\{-0.5,0,0.5\}$, and
$\Lambda\in\{300,800,2600,5000\}$ (60 cases), with the independent
disruption frequency and torus ratio and each
$f_{\rm RD}/f_{\rm tide}\in\{0.8,1,1.2\}$. This last ringdown frequency is
a controlled branch input, not a validation of the remnant/QNM model.
The native gates use CPU FP64 on both Torch and JAX.

## Mathematics and components

[Thompson et al. (2020), Table I and Sec. II.A step 6](https://arxiv.org/pdf/2002.08383v1)
define two tests: $H=[f_{\rm tide}\ge f_{\rm RD}]$ and
$B=[T>0]$. The four classes are

| $H$ | $B$ | Table I class |
| --- | --- | --- |
| true | false | non-disruptive, no torus |
| true | true | mildly disruptive, torus remnant |
| false | false | mildly disruptive, no torus remnant |
| false | true | disruptive, torus remnant |

The equality $f_{\rm tide}=f_{\rm RD}$ is on the $H$ side; a zero torus
ratio is on the no-torus side. The signed torus fit may be negative; the
carrier adapter $\max(0,T)$ leaves $B$ unchanged. This gate compares the
discrete branch only. Derivatives are undefined across either boundary and
are not accepted as a smooth-gradient claim.

## Implementation mapping

The independent scalar function will be
`diffgw_reference.nsbh_branch.classify_branch`, reusing the already separate
disruption and torus references only for composition inputs. Both native
`approximants.imrphenom_nsbh` modules will expose a small predicate helper
used by the production amplitude assembly. Neither native module imports the
scalar reference.

## Evidence and limitations

The acceptance criterion is **exact class agreement on every frozen case**;
there is no floating-point error tolerance for a discrete decision. The
independent disruption-root and torus/ISCO component gates separately check
their numerical inputs. This branch gate will not certify the remnant-derived
$f_{\rm RD}$, the fitted amplitude parameters selected by a branch, waveform
agreement with an external binary, or physical accuracy.

## Reproduce the checks

Run `JAX_ENABLE_X64=1 PYTHONPATH=src pytest -q
tests/test_nsbh_branch_reference.py` with reference, Torch, and JAX extras on
CPU. The validation receipt will record the exact command, dependency
versions, counts, skips, and failures.
