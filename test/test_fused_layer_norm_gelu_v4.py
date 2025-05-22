import torch
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu

torch.manual_seed(0)

def test_fused_layer_norm_gelu():
    M, N = 64, 256
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton implementation
    y_triton = fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch implementation
    layer_norm = torch.nn.LayerNorm(N, eps=eps).to('cuda').to(torch.float32)
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    x_norm = layer_norm(x)
    y_torch = 0.5 * x_norm * (1.0 + torch.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Compare
    torch.allclose(y_triton, y_torch, atol=1e-2, rtol=0.0)

test_fused_layer_norm_gelu()