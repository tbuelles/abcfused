import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_torch(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    x_normed = x_hat * weight + bias
    gelu_out = 0.5 * x_normed * (1.0 + torch.tanh(0.7978845608028654 * (x_normed + 0.044715 * x_normed * x_normed * x_normed)))
    return gelu_out


torch.manual_seed(0)
N_ELEMENTS = 128
N_COLS = 256
x = torch.randn(N_ELEMENTS, N_COLS, dtype=torch.float32, device='cuda')
weight = torch.randn(N_COLS, dtype=torch.float32, device='cuda')
bias = torch.randn(N_COLS, dtype=torch.float32, device='cuda')
eps = 1e-5

output_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)
output_torch = layer_norm_gelu_torch(x, weight, bias, eps=eps)

torch.testing.assert_close(output_triton, output_torch, atol=1e-3, rtol=1e-3)