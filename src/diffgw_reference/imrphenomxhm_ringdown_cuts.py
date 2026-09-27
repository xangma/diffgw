# SPDX-License-Identifier: MIT
"""Independent scalar XHM default ringdown cuts, Eq. (36) and Table 2.

García-Quirós et al., arXiv:2001.10914v2. QNM frequencies are supplied;
exceptional (3,2) cut adjustments are outside this default rule.
"""

import mpmath as mp


def cuts(ring, damp):
    """Return (amplitude, phase) cut dictionaries for positive QNM inputs."""
    modes = ("22", "21", "33", "32", "44")
    r = {mode: mp.mpf(ring[mode]) for mode in modes}
    d = {mode: mp.mpf(damp[mode]) for mode in modes}
    if any(value <= 0 for value in (*r.values(), *d.values())):
        raise ValueError("QNM ring and damping frequencies must be positive")
    amplitude = {
        "21": mp.mpf("0.75") * r["21"],
        "33": mp.mpf("0.95") * r["33"],
        "32": r["22"],
        "44": mp.mpf("0.90") * r["44"],
    }
    phase = {
        "21": r["21"] - d["21"],
        "33": r["33"] - d["33"],
        "32": r["22"] - mp.mpf("0.5") * d["22"],
        "44": r["44"] - d["44"],
    }
    return amplitude, phase
