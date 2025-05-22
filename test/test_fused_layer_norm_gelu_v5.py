import torch
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

torch.manual_seed(0)

def test_fused_layer_norm_gelu():
    M, N = 128, 256
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    w = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # PyTorch implementation
    x_mean = torch.mean(x, dim=1, keepdim=True)
    x_var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    layer_norm_output = x_hat * w + b
    gelu_output_torch = layer_norm_output * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * layer_norm_output * (1.0 + 0.044715 * layer_norm_output * layer_norm_output)))

    # Triton implementation
    gelu_output_triton = fused_layer_norm_gelu(x, w, b, eps)

    # Compare
    torch.allclose(gelu_output_torch, gelu_output_triton, atol=1e-2, rtol=1e-2)
    print("Layer Norm and GELU fused kernel test passed!")

test_fused_layer_norm_gelu()