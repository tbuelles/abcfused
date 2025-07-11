import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_pytorch(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    x_hat = x_hat * weight + bias
    output = x_hat * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))
    return output


torch.manual_seed(0)
N = 128
M = 784
x = torch.randn(N, M, device='cuda', dtype=torch.float32)
weight = torch.randn(M, device='cuda', dtype=torch.float32)
bias = torch.randn(M, device='cuda', dtype=torch.float32)

output_triton = fused_layer_norm_gelu(x, weight, bias)
output_pytorch = layer_norm_gelu_pytorch(x, weight, bias)

torch.allclose(output_triton, output_pytorch, atol=1e-3, rtol=1e-3)