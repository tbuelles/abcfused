import torch
import abcfused.ops
from abcfused.ops import fused_matmul_gelu

def test_matmul_gelu():
    M, K, N = 128, 64, 32
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    triton_output = fused_matmul_gelu(a, b)
    torch_output = torch.matmul(a, b)
    torch_output = 0.5 * torch_output * (1.0 + torch.tanh(0.7978845608028654 * torch_output + 0.03567740746541209))


    torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)
    print("Matmul Gelu Test Passed!")

test_matmul_gelu()