import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn as nn

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(variance + eps)
    x_norm = weight * x_norm + bias
    return x_norm * 0.5 * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x_norm + 0.044715 * x_norm**3)))


torch.manual_seed(42)
x = torch.randn(1, 2048, device='cuda', dtype=torch.float32)
weight = torch.randn(2048, device='cuda', dtype=torch.float32)
bias = torch.randn(2048, device='cuda', dtype=torch.float32)
eps = 1e-5

output_triton = fused_layer_norm_gelu(x, weight, bias, eps)
output_torch = layer_norm_gelu_ref(x, weight, bias, eps)

torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)