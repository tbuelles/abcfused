import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

torch.manual_seed(0)

def test_fused_layer_norm_gelu():
    N_ROWS = 128
    N_COLS = 2048
    x = torch.randn(N_ROWS, N_COLS, dtype=torch.float32, device='cuda')
    weight = torch.randn(N_COLS, dtype=torch.float32, device='cuda')
    bias = torch.randn(N_COLS, dtype=torch.float32, device='cuda')
    eps = 1e-5

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(N_COLS, eps=eps).to('cuda')
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    ref_out = layer_norm(x)
    ref_out = F.gelu(ref_out)

    # Triton fused kernel
    triton_out = fused_layer_norm_gelu(x, weight, bias, eps=eps)

    torch.allclose(triton_out, ref_out, rtol=1e-3, atol=1e-3)
    assert torch.allclose(triton_out, ref_out, rtol=1e-3, atol=1e-3)


test_fused_layer_norm_gelu()