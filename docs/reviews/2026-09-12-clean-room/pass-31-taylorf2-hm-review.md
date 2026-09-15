# Pass 31: clean-room TaylorF2_HM review

This review used only the current TorchWave implementation, its tests, the
published equations in García-Quirós et al., arXiv:2001.10914 (Appendix E/FPN),
and independently derived spin-weighted-harmonic polynomials. No LAL or PyCBC
source was inspected.

The (3,1), (3,2), and (4,3) leading terms and their common SPA normalization
are source-consistent. The (3,1) and (4,3) projection polynomials agree with
the standard spin-weight -2 Wigner-d basis, including the negative-m parity
projection signs. The existing frequency mapping uses
`v_m=(2*pi*M*f/m)^(1/3)` and the phase alignment constants are shared by
modes with the same m. Equal-mass zeros, validation, batching, float32/64,
device handling, and autograd behavior are covered by the focused tests.

One exact coefficient defect was found and fixed: the (3,3) amplitude ratio
must include the SPA velocity conversion, giving

`(3/4)*sqrt(5/7)*(3/2)^(5/6)*delta*v22`.

The former `(3/4)*sqrt(15/14)` omitted a factor `(3/2)^(1/3)`. The inventory
test now checks the absolute (3,3) and (4,3) ratios against the 22 mode, so a
shared wrong normalization cannot cancel in a mode-to-mode comparison.

Validation:

```text
13 passed: tests/test_taylorf2_hm.py tests/test_taylorf2_hm_mode_inventory.py
git diff --check: clean
```

The unrelated HM transition test change present in the working tree was left
untouched.
