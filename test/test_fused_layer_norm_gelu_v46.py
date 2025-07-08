import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(variance + eps)
    x_norm = x_norm * weight + bias
    gelu = 0.5 * x_norm * (1.0 + torch.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
    return gelu

torch.manual_seed(0)
N_ELEMENTS = 4096
ROW_SIZE = 1024
x = torch.randn(N_ELEMENTS, ROW_SIZE, device='cuda', dtype=torch.float32)
weight = torch.randn(ROW_SIZE, device='cuda', dtype=torch.float32)
bias = torch.randn(ROW_SIZE, device='cuda', dtype=torch.float32)
eps = 1e-5

output_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)
output_torch = layer_norm_gelu_ref(x, weight, bias, eps=eps)

torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)