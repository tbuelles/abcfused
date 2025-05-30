import torch
import abcfused.ops
import math

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_norm = x_norm * weight + bias
    return x_norm * 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x_norm + 0.044715 * x_norm ** 3)))

torch.manual_seed(0)
N_ROWS = 128
N_COLS = 256
x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
eps = 1e-5

# triton output
output_triton = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

# reference output
output_ref = layer_norm_gelu_ref(x, weight, bias, eps)

# compare
torch.allclose(output_triton, output_ref, atol=1e-3, rtol=1e-3)