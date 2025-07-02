import torch
import abcfused.ops
import pytest

def test_fused_matmul_softmax():
    torch.manual_seed(0)
    M, K, N = 128, 64, 32
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # Triton fused kernel
    C_triton = abcfused.ops.fused_matmul_softmax(A, B)

    # PyTorch equivalent
    C_matmul = torch.matmul(A, B)
    C_max = torch.max(C_matmul, dim=1, keepdim=True)[0]
    C_exp = torch.exp(C_matmul - C_max)
    C_sum = torch.sum(C_exp, dim=1, keepdim=True)
    C_softmax = C_exp / C_sum
    C_pytorch = C_softmax

    # Compare
    torch.set_printoptions(profile="full")
    assert torch.allclose(C_triton, C_pytorch, atol=1e-3, rtol=1e-3)