import torch
import torch.nn as nn
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu
import triton

class LayerNormGELU(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        self.eps = eps

    def forward(self, x):
        x = self.ln(x)
        x = torch.nn.functional.gelu(x)
        return x


def test_layer_norm_gelu():
    torch.manual_seed(42)
    N = 3
    M = 512
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    weight = torch.randn(M, device='cuda', dtype=torch.float32)
    bias = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5
    act_kind = 0  # 0=tanh, 1=fast_gelu (approximate)

    # Triton fused kernel
    y_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps, act_kind=act_kind)

    # PyTorch equivalent
    layer_norm = nn.LayerNorm(M, eps=eps).to('cuda').float()
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias

    y_torch = layer_norm(x)
    y_torch = torch.nn.functional.gelu(y_torch)


    torch.set_printoptions(precision=10)
    assert torch.allclose(y_triton, y_torch, atol=1e-4, rtol=1e-4)
    print("LayerNorm + GELU test passed!")

test_layer_norm_gelu()