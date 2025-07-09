import torch
import abcfused.ops

def test_fused_matmul_layer_norm():
    M, K, N = 128, 64, 32
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    W = torch.randn((N,), device='cuda', dtype=torch.float32)
    b = torch.randn((N,), device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton kernel
    C_triton, mean_triton, var_triton = abcfused.ops.fused_matmul_layer_norm(A, B, W, b, eps)

    # PyTorch equivalent
    C_torch = torch.matmul(A, B)
    mean_torch = torch.mean(C_torch, dim=1, keepdim=True)
    var_torch = torch.var(C_torch, dim=1, keepdim=True, unbiased=False)
    C_torch = (C_torch - mean_torch) / torch.sqrt(var_torch + eps)
    C_torch = C_torch * W + b

    assert torch.allclose(C_triton, C_torch, rtol=1e-3, atol=1e-3)