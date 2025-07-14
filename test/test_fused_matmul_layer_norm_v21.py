import torch
from abcfused.ops import fused_matmul_layer_norm

def test_fused_matmul_layer_norm():
    torch.manual_seed(0)
    M, K, N = 128, 64, 32
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    weight = torch.randn((N,), device='cuda', dtype=torch.float32)
    bias = torch.randn((N,), device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton fused kernel
    output_triton = fused_matmul_layer_norm(a, b, weight, bias, eps)

    # PyTorch equivalent
    matmul = torch.matmul(a, b)
    mean = torch.mean(matmul, dim=1, keepdim=True)
    var = torch.var(matmul, dim=1, keepdim=True, unbiased=False)
    norm = (matmul - mean) / torch.sqrt(var + eps)
    output_torch = norm * weight + bias

    assert torch.allclose(output_triton, output_torch, atol=1e-3, rtol=1e-3)

test_fused_matmul_layer_norm()