import torch
import torch.nn as nn
import triton
import triton.language as tl
from abcfused.ops import fused_layer_norm_gelu

class LayerNormGELU(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.layer_norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = self.layer_norm(x)
        return x * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


torch.manual_seed(0)
N, M = 4, 128
x = torch.randn(N, M, dtype=torch.float32, requires_grad=True, device='cuda')
weight = torch.randn(M, dtype=torch.float32, requires_grad=True, device='cuda')
bias = torch.randn(M, dtype=torch.float32, requires_grad=True, device='cuda')
eps = 1e-5

# triton version
y_triton = fused_layer_norm_gelu(x, weight, bias, eps)

# pytorch version
layer_norm_gelu = LayerNormGELU(M, eps).cuda()
layer_norm_gelu.weight.data = weight.clone()
layer_norm_gelu.bias.data = bias.clone()

y_torch = layer_norm_gelu(x)


torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2)