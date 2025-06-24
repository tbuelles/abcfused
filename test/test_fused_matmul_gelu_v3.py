import torch
import abcfused.ops
from abcfused.ops import fused_matmul_gelu

torch.manual_seed(0)

def test_matmul_gelu():
    M, K, N = 128, 64, 32
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)

    C_triton = fused_matmul_gelu(A, B)
    C_torch = torch.matmul(A, B)
    C_torch = torch.nn.functional.gelu(C_torch)

    torch.allclose(C_triton, C_torch, rtol=1e-3, atol=1e-3)
    assert torch.allclose(C_triton, C_torch, rtol=1e-3, atol=1e-3)

test_matmul_gelu()