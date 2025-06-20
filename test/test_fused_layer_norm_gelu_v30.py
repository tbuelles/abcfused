import torch
import abcfused.ops
import torch.nn.functional as F

def test_layer_norm_gelu():
    N_ROWS = 128
    N_COLS = 768
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(N_COLS, eps=eps).to('cuda').float()
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    x_norm = layer_norm(x)
    gelu_pytorch = F.gelu(x_norm)

    # Triton kernel
    gelu_triton = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

    # Compare
    torch.allclose(gelu_pytorch, gelu_triton, atol=1e-3, rtol=1e-3)
    assert torch.allclose(gelu_pytorch, gelu_triton, atol=1e-3, rtol=1e-3)