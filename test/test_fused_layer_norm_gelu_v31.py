import torch
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N_ROWS = 128
    N_COLS = 768
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    w = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    b = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton
    output_triton, mean_triton, variance_triton = fused_layer_norm_gelu.layer_norm_gelu(x, w, b, eps=eps)

    # PyTorch
    layer_norm = torch.nn.LayerNorm(N_COLS, eps=eps).to('cuda').to(torch.float32)
    layer_norm.weight.data = w
    layer_norm.bias.data = b
    output_torch = layer_norm(x)
    gelu = torch.nn.functional.gelu(output_torch)

    assert torch.allclose(output_triton, gelu, atol=1e-2, rtol=1e-2)

test_layer_norm_gelu()