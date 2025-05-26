import torch
from abcfused.ops import fused_matmul_layer_norm

def test_fused_matmul_layer_norm():
    torch.manual_seed(0)
    M, K, N = 128, 64, 256
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    W = torch.randn((N,), device='cuda', dtype=torch.float32)
    b = torch.randn((N,), device='cuda', dtype=torch.float32)
    eps = 1e-5

    C_triton = fused_matmul_layer_norm(A, B, W, b, eps)

    C_torch = torch.matmul(A, B)
    mean = torch.mean(C_torch, dim=1, keepdim=True)
    var = torch.var(C_torch, dim=1, keepdim=True, unbiased=False)
    C_torch = (C_torch - mean) / torch.sqrt(var + eps)
    C_torch = C_torch * W + b

    torch.testing.assert_close(C_triton, C_torch, rtol=1e-3, atol=1e-3)

test_fused_matmul_layer_norm()