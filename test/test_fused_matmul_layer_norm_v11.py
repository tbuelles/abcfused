import torch
import abcfused.ops
import pytest

def test_matmul_layer_norm():
    M, K, N = 64, 32, 64
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    weight = torch.randn(M, device='cuda', dtype=torch.float32)
    bias = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    C_triton = abcfused.ops.fused_matmul_layer_norm(A, B, weight, bias, eps)

    C_matmul = torch.matmul(A, B)
    mean = torch.mean(C_matmul, dim=1, keepdim=True)
    var = torch.var(C_matmul, dim=1, keepdim=True, unbiased=False)
    norm = (C_matmul - mean) / torch.sqrt(var + eps)
    C_torch = norm * weight[:, None] + bias[:, None]

    torch.testing.assert_close(C_triton, C_torch, rtol=1e-03, atol=1e-03)