import torch
import abcfused.ops
import triton

def test_fused_layer_norm_gelu():
    N = 2
    M = 1024
    dtype = torch.float32
    
    x = torch.randn(N, M, dtype=dtype, device='cuda')
    weight = torch.randn(M, dtype=dtype, device='cuda')
    bias = torch.randn(M, dtype=dtype, device='cuda')
    eps = 1e-5

    # Triton fused kernel
    y_triton = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    x_norm = torch.nn.functional.layer_norm(x, (M,), weight, bias, eps)
    y_torch = x_norm * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Compare
    torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)
    print("Test passed!")

test_fused_layer_norm_gelu()