import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N, M = 5, 64
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    w = torch.randn(M, device='cuda', dtype=torch.float32)
    b = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # PyTorch reference
    x_norm = torch.nn.functional.layer_norm(x, (M,), weight=w, bias=b, eps=eps)
    gelu = torch.nn.functional.gelu(x_norm)

    # Triton kernel
    y_triton = fused_layer_norm_gelu.layer_norm_gelu(x, w, b, eps=eps)

    # Compare
    torch.allclose(gelu, y_triton, atol=1e-2, rtol=1e-2)
    assert torch.allclose(gelu, y_triton, atol=1e-2, rtol=1e-2)