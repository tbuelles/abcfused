import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

def test_fused_layer_norm_gelu():
    N = 32
    M = 256
    eps = 1e-5

    x = torch.randn(N, M, dtype=torch.float32, device='cuda')
    weight = torch.randn(M, dtype=torch.float32, device='cuda')
    bias = torch.randn(M, dtype=torch.float32, device='cuda')

    # Triton fused kernel
    y_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(M, eps=eps).cuda()
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    y_torch_ln = layer_norm(x)
    y_torch = 0.5 * y_torch_ln * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (y_torch_ln + 0.044715 * y_torch_ln**3)))

    torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)

test_fused_layer_norm_gelu()