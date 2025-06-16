import torch
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(variance + eps)
    x_scaled = x_hat * weight + bias
    gelu_val = 0.5 * x_scaled * (1.0 + torch.tanh(torch.sqrt(2.0 / torch.pi) * (x_scaled + 0.044715 * x_scaled**3)))
    return gelu_val, mean.squeeze(), variance.squeeze()

torch.manual_seed(0)
N, D = 128, 256
x = torch.randn(N, D, device='cuda', dtype=torch.float32)
weight = torch.randn(D, device='cuda', dtype=torch.float32)
bias = torch.randn(D, device='cuda', dtype=torch.float32)
eps = 1e-5

# triton output
output_triton, mean_triton, var_triton = fused_layer_norm_gelu(x, weight, bias, eps)

# reference output
output_ref, mean_ref, var_ref = layer_norm_gelu_ref(x, weight, bias, eps)

assert torch.allclose(output_triton, output_ref, atol=1e-3, rtol=1e-3)
assert torch.allclose(mean_triton, mean_ref, atol=1e-3, rtol=1e-3)
assert torch.allclose(var_triton, var_ref, atol=1e-3, rtol=1e-3)