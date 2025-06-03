import torch
import torch.nn as nn
from abcfused.ops import fused_matmul_layer_norm

torch.manual_seed(0)

M, K, N = 64, 32, 128
a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)
w_norm = torch.randn(N, device='cuda', dtype=torch.float32)
b_norm = torch.randn(N, device='cuda', dtype=torch.float32)
eps = 1e-5

# Triton fused kernel
output_triton = fused_matmul_layer_norm(a, b, w_norm, b_norm, eps)

# PyTorch equivalent
c = torch.matmul(a, b)
layer_norm = nn.LayerNorm(N, eps=eps).to('cuda')
layer_norm.weight = torch.nn.Parameter(w_norm)
layer_norm.bias = torch.nn.Parameter(b_norm)
output_torch = layer_norm(c)

# Compare
torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)