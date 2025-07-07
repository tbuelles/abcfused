import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    device = "cuda"
    dtype = torch.float32
    N = 128
    D = 768
    x = torch.randn(N, D, device=device, dtype=dtype)
    weight = torch.randn(D, device=device, dtype=dtype)
    bias = torch.randn(D, device=device, dtype=dtype)
    eps = 1e-5

    # Triton fused kernel
    y_triton = fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(D, eps=eps).to(device, dtype)
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    y_torch = layer_norm(x)
    y_torch = 0.5 * y_torch * (1.0 + torch.tanh(0.7978845608028654 * (y_torch + 0.044715 * y_torch * y_torch * y_torch)))

    # Compare
    torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)

test_layer_norm_gelu()