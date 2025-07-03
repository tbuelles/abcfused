import torch
import torch.nn as nn
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu

class LayerNormGELU(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps)
        self.eps = eps
    def forward(self, x):
        x = self.ln(x)
        return x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

torch.manual_seed(0)
N, M = 16, 2048
x = torch.randn(N, M, device='cuda', dtype=torch.float32)
weight = torch.randn(M, device='cuda', dtype=torch.float32)
bias = torch.randn(M, device='cuda', dtype=torch.float32)
eps = 1e-5

# triton
output_triton = fused_layer_norm_gelu(x, weight, bias, eps)

# pytorch
layer_norm = nn.LayerNorm(M, eps=eps).cuda()
layer_norm.weight.data = weight
layer_norm.bias.data = bias
output_pytorch = layer_norm(x)
output_pytorch = output_pytorch * 0.5 * (1.0 + torch.tanh(0.7978845608 * (output_pytorch + 0.044715 * output_pytorch * output_pytorch * output_pytorch)))

torch.allclose(output_triton, output_pytorch, rtol=1e-3, atol=1e-3)