import torch
import fused

def test_layer_norm_gelu():
    M = 256
    N = 1024
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # reference
    x_norm = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
    expected = x_norm * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # triton
    actual = fused.ops.fused_layer_norm_gelu(x, weight, bias, eps)
    torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)
    assert torch.allclose(actual, expected, rtol=1e-2, atol=1e-2)

test_layer_norm_gelu()