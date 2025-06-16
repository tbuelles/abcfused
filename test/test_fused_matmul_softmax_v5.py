import torch
import abcfused.ops
from abcfused.ops import fused_matmul_softmax

torch.manual_seed(0)
M, K, N = 128, 64, 32

a = torch.randn((M, K), dtype=torch.float32, device='cuda')
b = torch.randn((K, N), dtype=torch.float32, device='cuda')

triton_output = fused_matmul_softmax(a, b)

matmul_output = torch.matmul(a, b)
max_vals = torch.max(matmul_output, dim=1, keepdim=True)[0]
exp_values = torch.exp(matmul_output - max_vals)
sum_exp = torch.sum(exp_values, dim=1, keepdim=True)
torch_output = exp_values / sum_exp

torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)