import torch
import abcfused.ops
from torch.nn import LayerNorm
import math

def reference_layer_norm_gelu(x, weight, bias, eps=1e-5):
    layer_norm = LayerNorm(x.shape[-1], eps=eps)
    layer_norm.weight = torch.nn.Parameter(weight)
    layer_norm.bias = torch.nn.Parameter(bias)
    x_norm = layer_norm(x)
    return x_norm * 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))


def test_layer_norm_gelu():
    N, M = 512, 256
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    weight = torch.randn(M, device='cuda', dtype=torch.float32)
    bias = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    triton_output = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)
    torch_output = reference_layer_norm_gelu(x, weight, bias, eps)

    torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)

test_layer_norm_gelu()