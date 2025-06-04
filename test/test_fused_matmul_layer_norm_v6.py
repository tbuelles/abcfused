import torch
import abcfused.ops
from abcfused.ops import fused_matmul_layer_norm

torch.manual_seed(42)

def test_fused_matmul_layer_norm():
    M, K, N = 128, 64, 32
    eps = 1e-5
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)

    # Triton fused kernel
    triton_output = fused_matmul_layer_norm(a, b, weight, bias, eps)

    # PyTorch equivalent
    torch_matmul = torch.matmul(a, b)
    mean = torch.mean(torch_matmul, dim=1, keepdim=True)
    var = torch.var(torch_matmul, dim=1, keepdim=True, unbiased=False)
    norm = (torch_matmul - mean) / torch.sqrt(var + eps)
    torch_output = norm * weight + bias

    torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)
    assert torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)

test_fused_matmul_layer_norm()