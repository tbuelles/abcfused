import torch
import torch.nn as nn
import triton.language as tl
from abcfused.ops import fused_layer_norm_dropout

torch.manual_seed(0)

def test_fused_layer_norm_dropout():
    N_ROWS = 256
    N_COLS = 128
    EPS = 1e-5
    P = 0.2

    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    scale = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)

    # Triton fused kernel
    output_fused = fused_layer_norm_dropout(x, scale, bias, EPS, P)

    # PyTorch equivalent
    layer_norm = nn.LayerNorm(N_COLS, eps=EPS).to('cuda').float()
    layer_norm.weight = nn.Parameter(scale)
    layer_norm.bias = nn.Parameter(bias)
    
    dropout = nn.Dropout(p=P)
    
    output_torch = dropout(layer_norm(x))

    # Compare
    torch.allclose(output_fused, output_torch, rtol=1e-2, atol=1e-2)
    assert torch.allclose(output_fused, output_torch, rtol=1e-2, atol=1e-2)