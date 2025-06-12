import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

torch.manual_seed(0)

def test_fused_layer_norm_gelu():
    N_ROW = 128
    N_COL = 256
    x = torch.randn(N_ROW, N_COL, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COL, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COL, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton fused kernel
    y_triton = fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(N_COL, eps=eps).cuda()
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    x_normed = layer_norm(x)
    y_torch = 0.5 * x_normed * (1.0 + torch.tanh(0.7978845608028654 * (x_normed + 0.044715 * x_normed * x_normed * x_normed)))

    torch.testing.assert_close(y_triton, y_torch, atol=1e-3, rtol=1e-3)