import torch
import abcfused.ops
import torch.nn.functional as F

def test_fused_layer_norm_gelu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    N = 128
    M = 768
    x = torch.randn(N, M, device=device, dtype=torch.float32)
    w = torch.randn(M, device=device, dtype=torch.float32)
    b = torch.randn(M, device=device, dtype=torch.float32)
    eps = 1e-5

    # PyTorch equivalent
    x_normalized = F.layer_norm(x, (M,), weight=w, bias=b, eps=eps)
    expected = x_normalized * 0.5 * (1.0 + torch.erf(x_normalized / torch.sqrt(torch.tensor(2.0))))

    # Triton kernel
    actual = abcfused.ops.fused_layer_norm_gelu(x, w, b, eps)

    # Compare
    torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)