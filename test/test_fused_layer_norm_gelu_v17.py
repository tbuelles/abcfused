import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_pytorch(x, weight, bias, eps=1e-5):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    x_var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - x_mean) / torch.sqrt(x_var + eps)
    x_scaled = x_norm * weight + bias
    gelu_output = 0.5 * x_scaled * (1 + torch.tanh(0.7978845608028654 * x_scaled * (1 + 0.044715 * x_scaled * x_scaled)))
    return gelu_output

torch.manual_seed(0)
N = 256
M = 64
x = torch.randn(M, N, device='cuda', dtype=torch.float32)
weight = torch.randn(N, device='cuda', dtype=torch.float32)
bias = torch.randn(N, device='cuda', dtype=torch.float32)
eps = 1e-5

output_triton = fused_layer_norm_gelu(x, weight, bias, eps)
output_pytorch = layer_norm_gelu_pytorch(x, weight, bias, eps)

torch.allclose(output_triton, output_pytorch, rtol=1e-03, atol=1e-03)