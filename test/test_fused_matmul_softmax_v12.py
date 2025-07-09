import torch
import abcfused.ops
from abcfused.ops import fused_matmul_softmax

def test_fused_matmul_softmax():
    torch.manual_seed(0)
    M, K, N = 64, 32, 64
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    triton_output = fused_matmul_softmax(a, b)
    torch_output = torch.softmax(torch.matmul(a, b), dim=-1)

    torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)