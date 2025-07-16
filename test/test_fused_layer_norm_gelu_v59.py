import torch
import fused.ops
import pytest

def test_fused_layer_norm_gelu():
    torch.manual_seed(0)
    N_ROWS, N_COLS = 128, 2048
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton kernel
    output_triton = fused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(N_COLS, eps=eps).to('cuda').to(torch.float32)
    layer_norm.weight = torch.nn.Parameter(weight)
    layer_norm.bias = torch.nn.Parameter(bias)

    output_torch = layer_norm(x)
    output_torch = torch.nn.functional.gelu(output_torch)

    # Compare
    torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2)