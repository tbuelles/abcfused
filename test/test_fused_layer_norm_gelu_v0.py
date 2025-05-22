import torch
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu

torch.manual_seed(0)

def test_layer_norm_gelu():
    N_ROWS = 128
    N_COLS = 2048
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # reference
    x_mean = torch.mean(x, dim=1, keepdim=True)
    x_var = torch.var(x, dim=1, keepdim=True)
    x_norm = (x - x_mean) / torch.sqrt(x_var + eps)
    x_norm = x_norm * weight + bias
    ref_out = 0.5 * x_norm * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x_norm + 0.044715 * x_norm**3)))

    # triton
    tri_out, _, _ = fused_layer_norm_gelu(x, weight, bias, eps)

    torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)