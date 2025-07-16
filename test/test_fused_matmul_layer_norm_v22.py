import torch
from abcfused.ops import fused_matmul_layer_norm

def test_matmul_layer_norm():
    M, K, N = 128, 64, 256
    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')
    bias = torch.randn(M, dtype=torch.float32, device='cuda')
    gamma = torch.randn(M, dtype=torch.float32, device='cuda')

    triton_output = fused_matmul_layer_norm(a, b, bias, gamma)

    matmul_output = torch.matmul(a, b)
    mean = torch.mean(matmul_output, dim=1, keepdim=True)
    variance = torch.var(matmul_output, dim=1, keepdim=True, unbiased=False)
    layer_norm_output = (matmul_output - mean) / torch.sqrt(variance + 1e-5) * gamma[:, None] + bias[:, None]

    torch.allclose(triton_output, layer_norm_output, rtol=1e-03, atol=1e-03)
    assert torch.allclose(triton_output, layer_norm_output, rtol=1e-03, atol=1e-03)

test_matmul_layer_norm()