import torch
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu

torch.manual_seed(0)

def test_layer_norm_gelu():
    N_ROWS = 128
    N_COLS = 768
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # reference
    x_mean = torch.mean(x, dim=1, keepdim=True)
    x_var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    x_norm = x_hat * weight + bias
    ref_out = x_norm * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # triton
    tri_out = fused_layer_norm_gelu(x, weight, bias, eps)

    torch.testing.assert_close(ref_out, tri_out, rtol=1e-3, atol=1e-3)

    print("✅ layer_norm + gelu test passed!")

test_layer_norm_gelu()