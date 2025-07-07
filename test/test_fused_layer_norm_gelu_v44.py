import torch
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_pytorch(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_scaled = x_norm * weight + bias
    gelu_val = 0.5 * x_scaled * (1.0 + torch.tanh(0.7978845608028654 * (x_scaled + 0.044715 * x_scaled * x_scaled * x_scaled)))
    return gelu_val


torch.manual_seed(0)
N, M = 4, 64
x = torch.randn(N, M, dtype=torch.float32, device='cuda')
weight = torch.randn(M, dtype=torch.float32, device='cuda')
bias = torch.randn(M, dtype=torch.float32, device='cuda')
eps = 1e-5

# triton result
y_triton = fused_layer_norm_gelu.layer_norm_gelu(x, weight, bias, eps=eps)

# torch result
y_torch = layer_norm_gelu_pytorch(x, weight, bias, eps=eps)

torch.allclose(y_triton, y_torch, rtol=1e-3, atol=1e-3)