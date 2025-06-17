import torch
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(variance + eps)
    x_norm = x_norm * weight + bias
    return x_norm * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

torch.manual_seed(0)
N = 4
M = 256
x = torch.randn(N, M, dtype=torch.float32, requires_grad=False).cuda()
weight = torch.randn(M, dtype=torch.float32, requires_grad=False).cuda()
bias = torch.randn(M, dtype=torch.float32, requires_grad=False).cuda()
output_triton = fused_layer_norm_gelu(x, weight, bias)
output_torch = layer_norm_gelu_ref(x, weight, bias)
torch.testing.assert_close(output_triton, output_torch, rtol=1e-3, atol=1e-3)