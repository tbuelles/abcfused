import torch
import abcfused.ops
from abcfused.ops import fused_matmul_gelu

torch.manual_seed(0)

M = 256
K = 128
N = 64

a = torch.randn((M, K), device='cuda', dtype=torch.float32)
b = torch.randn((K, N), device='cuda', dtype=torch.float32)

# triton implementation
triton_output = fused_matmul_gelu(a, b)

# pytorch implementation
torch_output = torch.matmul(a, b)
torch_output = torch_output * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (torch_output + 0.044715 * torch_output * torch_output * torch_output)))

# compare
torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)