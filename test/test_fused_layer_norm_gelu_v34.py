import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, M = 64, 512
    x = torch.randn(N, M, device=device, dtype=torch.float32)
    w = torch.randn(M, device=device, dtype=torch.float32)
    b = torch.randn(M, device=device, dtype=torch.float32)
    eps = 1e-5

    # --- triton ---
    y_triton = fused_layer_norm_gelu.layer_norm_gelu(x, w, b, eps)

    # --- torch ---
    x_norm = torch.nn.functional.layer_norm(x, (M,), w, b, eps)
    y_torch = x_norm * 0.5 * (1.0 + torch.erf(x_norm / (2.0**0.5)))

    torch.allclose(y_triton, y_torch, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y_triton, y_torch, rtol=1e-3, atol=1e-3)

test_layer_norm_gelu()