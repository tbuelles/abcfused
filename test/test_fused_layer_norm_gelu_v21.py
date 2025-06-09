import torch
import fused.ops
from torch.nn import functional as F

def test_fused_layer_norm_gelu():
    N = 3
    M = 64
    dtype = torch.float32
    device = 'cuda'

    x = torch.randn(N, M, dtype=dtype, device=device)
    w = torch.randn(M, dtype=dtype, device=device)
    b = torch.randn(M, dtype=dtype, device=device)
    eps = 1e-5

    # PyTorch reference
    x_mean = torch.mean(x, dim=1, keepdim=True)
    x_var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    x_norm = x_hat * w + b
    ref_out = x_norm * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Triton kernel
    tri_out, tri_mean, tri_var = fused.ops.fused_layer_norm_gelu(x, w, b, eps=eps)
    ref_mean = torch.mean(x, dim=1)
    ref_var = torch.var(x, dim=1, unbiased=False)


    assert torch.allclose(tri_out, ref_out, atol=1e-3, rtol=1e-3)
    assert torch.allclose(tri_mean, ref_mean, atol=1e-3, rtol=1e-3)
    assert torch.allclose(tri_var, ref_var, atol=1e-3, rtol=1e-3)

test_fused_layer_norm_gelu()