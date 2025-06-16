import torch
from abcfused.ops import fused_matmul_layer_norm

def test_fused_matmul_layer_norm():
    torch.manual_seed(0)
    M, K, N = 128, 64, 256
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    bias = torch.randn((M,), device='cuda', dtype=torch.float32)
    gamma = torch.randn((M,), device='cuda', dtype=torch.float32)
    var = torch.rand((), device='cuda', dtype=torch.float32)

    # Triton
    triton_output = fused_matmul_layer_norm(a, b, bias, gamma, var)

    # PyTorch equivalent
    matmul_output = torch.matmul(a, b)
    mean = torch.mean(matmul_output, dim=1, keepdim=True)
    variance = torch.var(matmul_output, dim=1, keepdim=True, unbiased=False) # unbiased=False is crucial here
    inv_std = torch.rsqrt(variance + var)
    pytorch_output = (matmul_output - mean) * (gamma[:, None] * inv_std) + bias[:, None]

    torch.testing.assert_close(triton_output, pytorch_output, rtol=1e-03, atol=1e-03)