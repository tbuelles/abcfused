import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

torch.manual_seed(0)

def test_layer_norm_gelu():
    N, M = 10, 128
    eps = 1e-5
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    w = torch.randn(M, device='cuda', dtype=torch.float32)
    b = torch.randn(M, device='cuda', dtype=torch.float32)

    # reference
    x_mean = torch.mean(x, dim=1, keepdim=True)
    x_var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    x_norm = (x - x_mean) / torch.sqrt(x_var + eps)
    y_ref = x_norm * w + b
    y_ref = y_ref * 0.5 * (1.0 + torch.erf(y_ref / 1.41421356237))

    # triton
    y_triton = fused_layer_norm_gelu.layer_norm_gelu(x, w, b, eps)

    torch.testing.assert_close(y_ref, y_triton, rtol=1e-3, atol=1e-3)

test_layer_norm_gelu()