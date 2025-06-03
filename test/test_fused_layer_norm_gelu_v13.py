import torch
import abcfused.ops

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N = 128
    M = 2048
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    w = torch.randn(M, device='cuda', dtype=torch.float32)
    b = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # PyTorch implementation
    layer_norm = nn.LayerNorm(M, eps=eps).to('cuda').to(torch.float32)
    layer_norm.weight.data = w
    layer_norm.bias.data = b
    x_norm = layer_norm(x)
    gelu = nn.GELU()
    expected = gelu(x_norm)

    # Triton implementation
    actual = abcfused.ops.fused_layer_norm_gelu(x, w, b, eps)

    torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)

test_layer_norm_gelu()