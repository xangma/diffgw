# SEOBNRv4 Hamiltonian same-state diagnostic

This is a development-only component diagnostic, not a release qualification.
The [plan](hamiltonian_same_state_plan.json) pins the annotated
`lalsimulation-v6.2.1` tag object (`5154f04e470d1cc48d6f61193af728b5692474ab`),
its peeled commit, source-file hashes, selected public-header hashes, and the
installed LALSuite wheel and linked-library hashes. The
[result](hamiltonian_same_state.json) records
the compiler invocation, frozen states, and measured values.

The [driver](../../diagnose_hamiltonian_same_state.py) uses `git show
<tag-object>:<path>` to extract the exact pinned `LALSimIMRSpinEOBHamiltonian.c`
and its three private headers into a temporary directory. It checks their
SHA-256 hashes against the plan. The compiler puts this directory first on its
include path and runs with `-H`; the driver checks the compiler's include trace
for all four paths. Four selected public LAL headers in the supplied merged
header tree must also match blobs read from that tag. The source checkout's
working-tree files and HEAD are never compilation inputs.

The [adapter](../../reference/hamiltonian_v4_source_wheel_adapter.c) uses the
pinned source's static `XLALSimIMRCalculateSpinEOBHCoeffs` initializer with
version 4, matching the non-tidal aligned-spin setup in pinned
`LALSimIMRSpinAlignedEOB.c` (coefficient setup around line 1445). It computes
the source generic Hamiltonian and calls the **wheel's** exported
`XLALSimInspiralEOBPAHamiltonianWrapper` with the same coefficient object,
spin dictionary, and explicit `(r, prstar, pphi)` state. The wheel wrapper's
source is `LALSimInspiralEOBPostAdiabatic.c` around line 1205; `analyticFlag=0`
selects the generic Hamiltonian and the wrapper sets tortoise flag 1. The
driver verifies the symbol in the wheel dylib with `nm -gU`, checks the adapter
imports it with `nm -u`, and confirms its Mach-O binding is specifically
`(from liblalsimulation.37)` with `nm -m`. This call does not resolve to the
compiled source implementation.

The macOS build uses Apple Clang with `-dynamiclib -O2 -H`, the temporary
pinned private include directory, the checked merged LAL header tree, and GSL
headers. It links absolute paths to the pinned wheel's
`liblalsimulation.37.dylib`, `liblal.20.dylib`, `libgsl.28.dylib`, and
`libgslcblas.0.dylib`. `-Wl,-headerpad_max_install_names` leaves room to
replace their wheel `/DLC/lal/.dylibs/...` install names with absolute paths
using `install_name_tool -change`. The driver confirms the final
`liblalsimulation` dependency using `otool -L`. All compiled files remain in a
temporary directory and are deleted after the diagnostic.

For four moderate-spin early-inspiral states and four strong aligned-spin
states, source and wheel `H_real/nu` agree bitwise (`max |difference| = 0`).
The native implementation differs from the wheel by at most
`4.1849154364e-08` (`1.0185538030e-08` relative) in the moderate-spin set.
In the strong-spin set, the maximum is `2.9037626590e-05`
(`7.1788089835e-06` relative) at `r = 3.7444098416`. The result contains
all eight values and calibration coefficients.

The wheel's private v4 coefficient initializer is not exported. The adapter
therefore initializes coefficients from the exact pinned source and verifies
that the wheel's exported Hamiltonian consumes them identically at the tested
states. It does not prove the wheel's private initialization path, all
Hamiltonian branches, or whole-waveform equivalence. The increasing native
strong-spin difference is attributed at these eight states by the separate
[two-factor convention diagnostic](hamiltonian_source_conventions.json): the
historical logarithmic derivative and the source header's shorter calibration
decimals together reduce it to floating-point noise. Both conventions remain
development-only pending full-dynamics and waveform validation.

Reproduce from this worktree using `PYTHONPATH=src:. python -m
dev.native_rom.diagnose_hamiltonian_same_state --source-root
/Users/xangma/repos/lalsuite-git --headers
/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include --gsl-include
/usr/local/include --json-out /tmp/hamiltonian-same-state.json` with the
merged pinned headers and GSL headers available at those paths.
