import torch
from abcfused.ops import fused_matmul_gelu

torch.manual_seed(0)

M, K, N = 256, 128, 256
A = torch.randn((M, K), device='cuda', dtype=torch.float32)
B = torch.randn((K, N), device='cuda', dtype=torch.float32)

triton_output = fused_matmul_gelu(A, B)
torch_output = torch.matmul(A, B)
torch_output = torch.nn.functional.gelu(torch_output)

torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)