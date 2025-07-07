import torch
import abcfused.ops

def test_fused_matmul_layer_norm():
    M = 64
    N = 64
    K = 64
    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')
    bias = torch.randn((N,), dtype=torch.float32, device='cuda')
    gamma = torch.randn((N,), dtype=torch.float32, device='cuda')
    eps = 1e-5

    # Triton fused kernel
    triton_output = abcfused.ops.fused_matmul_layer_norm(a, b, bias, gamma, eps)

    # PyTorch equivalent
    matmul_output = torch.matmul(a, b)
    matmul_output = matmul_output + bias
    mean = torch.mean(matmul_output, dim=1, keepdim=True)
    variance = torch.var(matmul_output, dim=1, keepdim=True, unbiased=False)
    layer_norm_output = (matmul_output - mean) / torch.sqrt(variance + eps)
    layer_norm_output = layer_norm_output * gamma

    assert torch.allclose(triton_output, layer_norm_output, rtol=1e-3, atol=1e-3)

test_fused_matmul_layer_norm()