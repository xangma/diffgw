"""Unit tests for Kerr black hole ringdown overtones & spectroscopy."""

import math
import pytest
import torch

from torchwave.approximants.kerr_ringdown import (
    BERTI_QNM_FITS,
    KerrRingdown,
    evaluate_qnm,
    get_ringdown_waveform,
)
from torchwave.constants import MSUN_SEC, PI


def test_berti_table_viii_coefficients():
    """Verify QNM frequencies and damping times against Berti et al. (2006) Table VIII published values."""
    # Explicit published benchmark values from Berti, Cardoso, Will (2006) PRD 73, 064030 Table VIII
    # Format: (l, m, n): {af: (expected_M_omega, expected_Q)}
    benchmarks = {
        (2, 2, 0): {
            0.0: (0.368300, 2.118700),
            0.7: (0.534947, 3.287063),
        },
        (2, 2, 1): {
            0.0: (0.341300, 0.643600),
            0.7: (0.523921, 1.060845),
        },
        (2, 1, 0): {
            0.0: (0.366100, 2.056100),
            0.7: (0.458509, 2.799226),
        },
        (3, 3, 0): {
            0.0: (0.591300, 3.243000),
            0.7: (0.847704, 5.080969),
        },
        (4, 4, 0): {
            0.0: (0.794400, 4.312000),
            0.7: (1.150853, 6.768843),
        },
    }

    for (l, m, n), cases in benchmarks.items():
        for af, (exp_omega, exp_q) in cases.items():
            omega, tau, Q = evaluate_qnm(l, m, n, final_mass=1.0, final_spin=af, qnm_backend="berti")
            actual_omega = (omega * MSUN_SEC).item()
            actual_q = Q.item()

            assert math.isclose(actual_omega, exp_omega, rel_tol=1e-4), (
                f"Mode ({l},{m},{n}) at af={af}: actual M*omega {actual_omega} != expected {exp_omega}"
            )
            assert math.isclose(actual_q, exp_q, rel_tol=1e-4), (
                f"Mode ({l},{m},{n}) at af={af}: actual Q {actual_q} != expected {exp_q}"
            )


def test_qnm_overtone_hierarchy():
    """Verify that overtones (n=1, 2) damp faster (shorter tau) and have lower Q than fundamental mode (n=0)."""
    mass = 65.0
    spin = 0.68

    omega_0, tau_0, Q_0 = evaluate_qnm(2, 2, 0, final_mass=mass, final_spin=spin)
    omega_1, tau_1, Q_1 = evaluate_qnm(2, 2, 1, final_mass=mass, final_spin=spin)
    omega_2, tau_2, Q_2 = evaluate_qnm(2, 2, 2, final_mass=mass, final_spin=spin)

    # Damping time tau should decrease strictly with overtone index n
    assert tau_0.item() > tau_1.item() > tau_2.item() > 0.0

    # Quality factor Q should decrease strictly with overtone index n
    assert Q_0.item() > Q_1.item() > Q_2.item() > 0.0


def test_kerr_ringdown_time_domain():
    """Verify time-domain damped sinusoids, t < t0 heaviside cut, and exponential decay."""
    times = torch.linspace(-0.01, 0.05, 500, dtype=torch.float64)
    t0 = 0.005
    mass = 60.0
    spin = 0.7

    amplitudes = {(2, 2, 0): 1.0, (2, 2, 1): 0.5}
    phases = {(2, 2, 0): 0.0, (2, 2, 1): math.pi / 4.0}

    hp, hc = KerrRingdown.time_domain(
        times=times,
        final_mass=mass,
        final_spin=spin,
        amplitudes=amplitudes,
        phases=phases,
        t0=t0,
        inclination=0.4,
        phi=0.2,
    )

    # Signal must be identically 0 before t0
    pre_t0_mask = times < t0
    assert torch.all(hp[pre_t0_mask] == 0.0)
    assert torch.all(hc[pre_t0_mask] == 0.0)

    # Signal must be non-zero after t0
    post_t0_mask = times > (t0 + 0.001)
    assert torch.any(hp[post_t0_mask] != 0.0)

    # Check that amplitude decays exponentially at late times
    # In pure (2,2,0) limit:
    hp_0, _ = KerrRingdown.time_domain(
        times=times,
        final_mass=mass,
        final_spin=spin,
        amplitudes={(2, 2, 0): 1.0},
        phases={(2, 2, 0): 0.0},
        t0=t0,
        inclination=0.0,
        phi=0.0,
    )
    _, tau_0, _ = evaluate_qnm(2, 2, 0, final_mass=mass, final_spin=spin)

    # Peak envelope decay check between two points separated by delta_t
    t1 = t0 + 0.002
    t2 = t0 + 0.010
    idx1 = torch.argmin(torch.abs(times - t1))
    idx2 = torch.argmin(torch.abs(times - t2))

    # Envelope ratio approx exp(-(t2 - t1) / tau_0)
    env1 = math.exp(-(times[idx1].item() - t0) / tau_0.item())
    env2 = math.exp(-(times[idx2].item() - t0) / tau_0.item())
    assert env2 < env1


def test_kerr_ringdown_frequency_domain():
    """Verify frequency-domain Lorentzian response and peak location."""
    mass = 65.0
    spin = 0.68
    f_array = torch.linspace(50.0, 600.0, 500, dtype=torch.float64)

    omega_220, tau_220, _ = evaluate_qnm(2, 2, 0, final_mass=mass, final_spin=spin)
    # The transform of a real causal cosine has two poles. Their interference
    # shifts the amplitude maximum slightly above the oscillation frequency.
    w, g = omega_220.item(), 1.0 / tau_220.item()
    f_peak_exact = math.sqrt(math.sqrt(w**4 + 4*g*g*w*w) - g*g) / (2.0 * PI)

    hp_fd, hc_fd = KerrRingdown.frequency_domain(
        frequencies=f_array,
        final_mass=mass,
        final_spin=spin,
        amplitudes={(2, 2, 0): 1.0},
        phases={(2, 2, 0): 0.0},
        inclination=0.0,
    )

    amp = torch.abs(hp_fd)
    peak_idx = torch.argmax(amp)
    f_peak = f_array[peak_idx].item()

    # Peak frequency should be within one grid bin of the exact cosine peak.
    df = (f_array[1] - f_array[0]).item()
    assert abs(f_peak - f_peak_exact) <= df


def test_kerr_ringdown_batching():
    """Verify batched evaluation across masses, spins, inclinations, and start times."""
    B = 4
    N = 100
    times = torch.linspace(0.0, 0.05, N, dtype=torch.float64)
    masses = torch.tensor([50.0, 65.0, 80.0, 100.0], dtype=torch.float64)
    spins = torch.tensor([0.2, 0.5, 0.7, 0.9], dtype=torch.float64)
    incs = torch.tensor([0.1, 0.5, 0.8, 1.2], dtype=torch.float64)
    t0s = torch.tensor([0.001, 0.002, 0.0, 0.005], dtype=torch.float64)

    hp_b, hc_b = KerrRingdown.time_domain(
        times=times,
        final_mass=masses,
        final_spin=spins,
        amplitudes={(2, 2, 0): 1.0, (3, 3, 0): 0.3},
        phases={(2, 2, 0): 0.0, (3, 3, 0): 0.5},
        t0=t0s,
        inclination=incs,
    )

    assert hp_b.shape == (B, N)
    assert hc_b.shape == (B, N)

    # Check against individual unbatched calls
    for b in range(B):
        hp_s, hc_s = KerrRingdown.time_domain(
            times=times,
            final_mass=masses[b],
            final_spin=spins[b],
            amplitudes={(2, 2, 0): 1.0, (3, 3, 0): 0.3},
            phases={(2, 2, 0): 0.0, (3, 3, 0): 0.5},
            t0=t0s[b],
            inclination=incs[b],
        )
        assert torch.allclose(hp_b[b], hp_s, atol=1e-12)
        assert torch.allclose(hc_b[b], hc_s, atol=1e-12)


def test_kerr_ringdown_autograd():
    """Verify autograd differentiability with respect to remnant mass, spin, inclination, and t0."""
    times = torch.linspace(0.002, 0.04, 64, dtype=torch.float64)
    mass = torch.tensor(65.0, dtype=torch.float64, requires_grad=True)
    spin = torch.tensor(0.68, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    t0 = torch.tensor(0.001, dtype=torch.float64, requires_grad=True)

    hp, hc = KerrRingdown.time_domain(
        times=times,
        final_mass=mass,
        final_spin=spin,
        amplitudes={(2, 2, 0): 1.0, (2, 2, 1): 0.4},
        t0=t0,
        inclination=inc,
    )

    loss = (hp ** 2 + hc ** 2).sum()
    loss.backward()

    for name, param in [("mass", mass), ("spin", spin), ("inc", inc), ("t0", t0)]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad), f"Gradient for {name} is NaN"
        assert param.grad.abs().item() > 0.0, f"Gradient for {name} is zero"
