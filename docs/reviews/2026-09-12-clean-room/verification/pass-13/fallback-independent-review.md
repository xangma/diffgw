# Independent Luna review of amplitude fallback constraints

The second Luna agent independently checked García-Quirós Sec. V.A.3,
Eq. (5.3), and Tables IV--VI without reference outputs or runtime edits.
Parent reviewed the same primary definitions.

- Default: four amplitude values and two endpoint derivatives determine the
  inverse quintic denominator.
- Parameter veto: removing both interior values leaves the endpoint Hermite
  cubic, preserving two values and two derivatives.
- Denominator pole: removing endpoint derivatives preserves all four values
  and yields a value-only cubic. If the parameter veto already removed the
  interior values, removing derivatives leaves endpoint linear interpolation.
- A remaining unacceptable denominator requires further reduction toward the
  endpoint linear form, the stated minimum order. The reconstruction is
  selected over the whole intermediate interval for each intrinsic row.

The source does not prescribe pointwise switching among polynomials at each
frequency. The inherited 11-point sampling and 5% near-zero threshold are
still heuristics, not an exact polynomial root test or certified source rule.
