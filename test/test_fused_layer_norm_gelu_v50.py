import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_norm = x_norm * weight + bias
    return 0.5 * x_norm * (1.0 + torch.tanh(0.7978845608028654 * x_norm * (1.0 + 0.044715 * x_norm * x_norm)))

torch.manual_seed(0)
N_ROW = 128
N_COL = 256
x = torch.randn(N_ROW, N_COL, device='cuda', dtype=torch.float32)
weight = torch.randn(N_COL, device='cuda', dtype=torch.float32)
bias = torch.randn(N_COL, device='cuda', dtype=torch.float32)
y_triton = fused_layer_norm_gelu(x, weight, bias)
y_torch = layer_norm_gelu_ref(x, weight, bias)
torch.allclose(y_triton, y_torch, rtol=1e-3, atol=1e-3)