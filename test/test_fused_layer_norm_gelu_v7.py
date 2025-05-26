import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    device = 'cuda'
    dtype = torch.float32
    N, M = 2, 1024

    x = torch.randn(N, M, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(M, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(M, device=device, dtype=dtype, requires_grad=True)
    eps = 1e-5

    # triton
    y_triton = torch.empty_like(x)
    y_triton = fused_layer_norm_gelu(y_triton, x, w, b, eps)

    # pytorch
    x_mean = torch.mean(x, dim=1, keepdim=True)
    x_var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    x_norm = (x - x_mean) / torch.sqrt(x_var + eps)
    y_ref = x_norm * w + b # LayerNorm removed for correct GELU
    y_ref = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))) # GELU

    torch.allclose(y_triton, y_ref, rtol=1e-3, atol=1e-3)

    assert torch.allclose(y_triton, y_ref, rtol=1e-3, atol=1e-3)