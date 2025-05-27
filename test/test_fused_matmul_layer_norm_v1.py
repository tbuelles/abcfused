import torch
from abcfused.ops import fused_matmul_layer_norm

def test_fused_matmul_layer_norm():
    torch.manual_seed(0)
    M, K, N = 64, 32, 64
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    W = torch.randn((M,), device='cuda', dtype=torch.float32)
    b = torch.randn((M,), device='cuda', dtype=torch.float32)
    eps = 1e-5

    C_triton = fused_matmul_layer_norm(A, B, W, b, eps)

    C_matmul = torch.matmul(A, B)
    mean = torch.mean(C_matmul, dim=1, keepdim=True)
    variance = torch.var(C_matmul, dim=1, keepdim=True)
    C_torch = (C_matmul - mean) / torch.sqrt(variance + eps) * W + b
    assert torch.allclose(C_triton, C_torch, rtol=1e-03, atol=1e-03)

test_fused_matmul_layer_norm()