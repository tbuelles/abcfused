import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

def layer_norm_gelu(x, weight, bias, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    x = (x - mean) / torch.sqrt(var + eps)
    x = x * weight + bias
    return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


torch.manual_seed(0)
N, M = 256, 512
x = torch.randn(N, M, device='cuda', dtype=torch.float32)
weight = torch.randn(M, device='cuda', dtype=torch.float32)
bias = torch.randn(M, device='cuda', dtype=torch.float32)
eps = 1e-5

# triton
output_triton, mean_triton, var_triton = fused_layer_norm_gelu(x, weight, bias, eps)

# pytorch
output_torch = layer_norm_gelu(x, weight, bias, eps)

torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)