import time
import torch
from torchwave.approximants.imrphenomd import TABLE_V_COEFFS

eta = torch.tensor(0.24, device='cuda', dtype=torch.float64)
chi_pn = torch.tensor(0.1, device='cuda', dtype=torch.float64)

# Method A (Current)
def method_a(eta, chi_pn):
    results = []
    for name, coeffs in TABLE_V_COEFFS.items():
        c = [torch.tensor(val, dtype=eta.dtype, device=eta.device) for val in coeffs]
        eta2 = eta * eta
        x = chi_pn - 1.0
        x2 = x * x
        x3 = x2 * x
        poly = (
            c[0] + c[1] * eta
            + x * (c[2] + c[3] * eta + c[4] * eta2)
            + x2 * (c[5] + c[6] * eta + c[7] * eta2)
            + x3 * (c[8] + c[9] * eta + c[10] * eta2)
        )
        results.append(poly)
    return results

# Method B (Floats)
def method_b(eta, chi_pn):
    results = []
    eta2 = eta * eta
    x = chi_pn - 1.0
    x2 = x * x
    x3 = x2 * x
    for name, c in TABLE_V_COEFFS.items():
        poly = (
            c[0] + c[1] * eta
            + x * (c[2] + c[3] * eta + c[4] * eta2)
            + x2 * (c[5] + c[6] * eta + c[7] * eta2)
            + x3 * (c[8] + c[9] * eta + c[10] * eta2)
        )
        results.append(poly)
    return results

# Method C (Single Matrix Multiplication)
param_names = list(TABLE_V_COEFFS.keys())
matrix_np = [TABLE_V_COEFFS[k] for k in param_names]
matrix_cuda = torch.tensor(matrix_np, device='cuda', dtype=torch.float64)

def method_c(eta, chi_pn):
    x = chi_pn - 1.0
    eta2 = eta * eta
    features = torch.stack([
        torch.ones_like(eta), eta,
        x, x * eta, x * eta2,
        x * x, (x * x) * eta, (x * x) * eta2,
        x * x * x, (x * x * x) * eta, (x * x * x) * eta2
    ], dim=-1)
    res = features @ matrix_cuda.T
    return res

# Warmup
for _ in range(10):
    method_a(eta, chi_pn)
    method_b(eta, chi_pn)
    method_c(eta, chi_pn)
torch.cuda.synchronize()

N = 100
t0 = time.perf_counter()
for _ in range(N):
    method_a(eta, chi_pn)
torch.cuda.synchronize()
t_a = (time.perf_counter() - t0) / N * 1000.0

t0 = time.perf_counter()
for _ in range(N):
    method_b(eta, chi_pn)
torch.cuda.synchronize()
t_b = (time.perf_counter() - t0) / N * 1000.0

t0 = time.perf_counter()
for _ in range(N):
    method_c(eta, chi_pn)
torch.cuda.synchronize()
t_c = (time.perf_counter() - t0) / N * 1000.0

print(f"Method A (current): {t_a:.4f} ms")
print(f"Method B (floats):  {t_b:.4f} ms (Speedup: {t_a/t_b:.1f}x)")
print(f"Method C (GEMM):    {t_c:.4f} ms (Speedup: {t_a/t_c:.1f}x)")

# Verify parity between A, B, and C
res_a = torch.stack(method_a(eta, chi_pn))
res_b = torch.stack(method_b(eta, chi_pn))
res_c = method_c(eta, chi_pn)
print(f"Diff A vs B: {(res_a - res_b).abs().max().item()}")
print(f"Diff A vs C: {(res_a - res_c).abs().max().item()}")
