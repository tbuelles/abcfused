import torch
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

torch.manual_seed(0)

def test_layer_norm_gelu():
    N_ROWS = 128
    N_COLS = 768
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # PyTorch implementation
    layer_norm = torch.nn.LayerNorm(N_COLS, eps=eps).to('cuda')
    layer_norm.weight = torch.nn.Parameter(weight)
    layer_norm.bias = torch.nn.Parameter(bias)
    output_torch = layer_norm(x)
    output_torch = F.gelu(output_torch)

    # Triton implementation
    output_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)

    # Compare
    torch.allclose(output_torch, output_triton, atol=1e-3, rtol=1e-3)
    assert torch.allclose(output_torch, output_triton, atol=1e-3, rtol=1e-3)

test_layer_norm_gelu()