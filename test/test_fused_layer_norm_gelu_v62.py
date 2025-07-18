import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn as nn

def layer_norm_gelu_torch(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    x_transformed = x_hat * weight + bias
    return x_transformed * 0.5 * (1.0 + torch.tanh(x_transformed * 0.7978845608 * (1.0 + 0.044715 * x_transformed * x_transformed)))

torch.manual_seed(0)
N, M = 128, 256
x = torch.randn(N, M, device='cuda', dtype=torch.float32)
weight = torch.randn(M, device='cuda', dtype=torch.float32)
bias = torch.randn(M, device='cuda', dtype=torch.float32)
eps = 1e-5

# triton
y_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)

# torch
y_torch = layer_norm_gelu_torch(x, weight, bias, eps=eps)

torch.allclose(y_triton, y_torch, rtol=1e-3, atol=1e-3)