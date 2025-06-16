import torch
import abcfused.ops
from abcfused.ops import fused_matmul_layer_norm

def test_fused_matmul_layer_norm():
    torch.manual_seed(42)
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    W = torch.randn(N, dtype=torch.float32, device='cuda')
    bias = torch.randn(N, dtype=torch.float32, device='cuda')
    eps = 1e-5

    C_triton, mean_triton, var_triton = fused_matmul_layer_norm(A, B, W, bias, eps)

    C_torch = torch.matmul(A, B)
    mean_torch = torch.mean(C_torch, dim=1, keepdim=True)
    var_torch = torch.var(C_torch, dim=1, keepdim=True, unbiased=False)
    C_torch = (C_torch - mean_torch) / torch.sqrt(var_torch + eps)
    C_torch = C_torch * W + bias
    mean_torch = mean_torch.squeeze(1)
    var_torch = var_torch.squeeze(1)

    assert torch.allclose(C_triton, C_torch, atol=1e-3, rtol=1e-3)
    assert torch.allclose(mean_triton, torch.mean(torch.matmul(A, B), dim=1), atol=1e-3, rtol=1e-3)
    assert torch.allclose(var_triton, torch.var(torch.matmul(A, B), dim=1, unbiased=False), atol=1e-3, rtol=1e-3)