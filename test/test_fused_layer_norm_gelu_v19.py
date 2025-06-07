import torch
import abcfused.ops
from torch.nn import LayerNorm
import torch.nn.functional as F

def test_layer_norm_gelu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_ROWS = 128
    N_COLS = 256
    x = torch.randn(N_ROWS, N_COLS, device=device, dtype=torch.float32)
    weight = torch.randn(N_COLS, device=device, dtype=torch.float32)
    bias = torch.randn(N_COLS, device=device, dtype=torch.float32)
    eps = 1e-5

    # reference
    layer_norm = LayerNorm(N_COLS).to(device)
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    layer_norm.eval()
    ref = layer_norm(x)
    ref = F.gelu(ref)
    # triton
    tri_out = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps=eps)

    torch.allclose(ref, tri_out, rtol=1e-3, atol=1e-3)
    assert torch.allclose(ref, tri_out, rtol=1e-3, atol=1e-3)