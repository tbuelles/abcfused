import torch
import abcfused.ops
from abcfused.ops import fused_matmul_layer_norm

torch.manual_seed(0)

def test_fused_matmul_layer_norm():
    M = 64
    N = 64
    K = 64
    eps = 1e-5
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # Triton fused kernel
    triton_output = fused_matmul_layer_norm(a, b, eps)

    # PyTorch equivalent
    matmul_output = torch.matmul(a, b)
    mean = torch.mean(matmul_output, dim=1, keepdim=True)
    variance = torch.var(matmul_output, dim=1, keepdim=True, unbiased=False)
    pytorch_output = (matmul_output - mean) / torch.sqrt(variance + eps)

    torch.allclose(triton_output, pytorch_output, atol=1e-2, rtol=1e-2)
    assert torch.allclose(triton_output, pytorch_output, atol=1e-2, rtol=1e-2)

test_fused_matmul_layer_norm()