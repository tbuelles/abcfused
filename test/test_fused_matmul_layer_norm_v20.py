import torch
import abcfused.ops
from abcfused.ops import fused_matmul_layer_norm

torch.manual_seed(0)
M, K, N = 128, 64, 256
A = torch.randn((M, K), device='cuda', dtype=torch.float32)
B = torch.randn((K, N), device='cuda', dtype=torch.float32)
W = torch.randn((N,), device='cuda', dtype=torch.float32)
b = torch.randn((N,), device='cuda', dtype=torch.float32)
eps = 1e-5

# Triton
C_triton = fused_matmul_layer_norm(A, B, W, b, eps)

# PyTorch
C_matmul = torch.matmul(A, B)
layer_norm = torch.nn.LayerNorm(N, eps=eps, elementwise_affine=True).to('cuda').float()
layer_norm.weight.data = W
layer_norm.bias.data = b
C_torch = layer_norm(C_matmul)

torch.allclose(C_triton, C_torch, rtol=1e-3, atol=1e-3)