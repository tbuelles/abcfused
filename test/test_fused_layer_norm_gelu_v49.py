import torch
import torch.nn as nn
from abcfused.ops import fused_layer_norm_gelu

torch.manual_seed(42)

def test_fused_layer_norm_gelu():
    N, M = 64, 256
    x = torch.randn(N, M, device="cuda", dtype=torch.float32)
    weight = torch.randn(M, device="cuda", dtype=torch.float32)
    bias = torch.randn(M, device="cuda", dtype=torch.float32)
    eps = 1e-5

    # Triton version
    output_triton = fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch version
    layer_norm = nn.LayerNorm(M, eps=eps).cuda()
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    output_torch = layer_norm(x)
    output_torch = output_torch * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (output_torch + 0.044715 * output_torch * output_torch * output_torch)))
    

    torch.allclose(output_triton, output_torch, atol=1e-3, rtol=1e-3)
    assert torch.allclose(output_triton, output_torch, atol=1e-3, rtol=1e-3)

test_fused_layer_norm_gelu()