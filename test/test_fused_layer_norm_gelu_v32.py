import torch
import torch.nn as nn
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    N_ROW = 128
    N_COL = 2048
    x = torch.randn(N_ROW, N_COL, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COL, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COL, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton fused kernel
    output_triton = fused_layer_norm_gelu.layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    layer_norm = nn.LayerNorm(N_COL, eps=eps).to('cuda').float()
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    output_torch = layer_norm(x)
    output_torch = output_torch * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * output_torch * (1.0 + 0.044715 * output_torch * output_torch)))

    assert torch.allclose(output_triton, output_torch, atol=1e-3, rtol=1e-3)
    print("LayerNorm + GELU fusion test passed!")

test_layer_norm_gelu()