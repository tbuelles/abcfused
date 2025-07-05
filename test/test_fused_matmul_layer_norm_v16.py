import torch
import triton
import triton.language as tl
from abcfused.ops import fused_matmul_layer_norm

torch.manual_seed(0)
def test_matmul_layer_norm():
    M = 128
    N = 64
    K = 32
    a = torch.randn((M, K), device='cuda', dtype=torch.float32, requires_grad=False)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32, requires_grad=False)
    eps = 1e-5

    # Triton version
    triton_output, triton_mean = fused_matmul_layer_norm(a, b, eps=eps)

    # PyTorch version
    torch_output = torch.matmul(a, b)
    mean = torch.mean(torch_output, dim=1, keepdim=True)
    var = torch.var(torch_output, dim=1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    torch_output = (torch_output - mean) / std

    assert torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)
    #Verification with respect to the stored mean values
    assert torch.allclose(triton_mean, mean.flatten(), atol=1e-3, rtol=1e-3)