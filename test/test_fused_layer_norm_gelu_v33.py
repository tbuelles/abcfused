import torch
import torch.nn as nn
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu


class LayerNormGELU(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        self.eps = eps

    def forward(self, x):
        x = self.ln(x)
        return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def test_layer_norm_gelu():
    torch.manual_seed(0)
    N, M = 4, 256
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    weight = torch.randn(M, device='cuda', dtype=torch.float32)
    bias = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # reference
    layer_norm_gelu = LayerNormGELU(M, eps=eps).to('cuda')
    layer_norm_gelu.ln.weight.data = weight
    layer_norm_gelu.ln.bias.data = bias
    torch_output = layer_norm_gelu(x)

    # triton
    triton_output = fused_layer_norm_gelu(x, weight, bias, eps)

    torch.allclose(torch_output, triton_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(torch_output, triton_output, atol=1e-3, rtol=1e-3)
    print("✅ Triton and Torch LayerNorm+GELU match!")

test_layer_norm_gelu()