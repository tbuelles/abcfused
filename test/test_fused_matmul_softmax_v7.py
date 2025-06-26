import torch
from abcfused.ops import fused_matmul_softmax

torch.manual_seed(0)

M, K, N = 128, 64, 32
a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)

# triton version
c_triton = fused_matmul_softmax(a, b)

# torch version
c_torch = torch.matmul(a, b)
c_torch = torch.softmax(c_torch, dim=1)

# compare
torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2)