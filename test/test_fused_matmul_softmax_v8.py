import torch
import abcfused.ops
from abcfused.ops import fused_matmul_softmax

torch.manual_seed(0)

def test_matmul_softmax():
    M, K, N = 128, 64, 32
    a = torch.randn((M, K), device='cuda', dtype=torch.float32, requires_grad=False)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32, requires_grad=False)

    # triton implementation
    triton_output = fused_matmul_softmax(a, b)

    # pytorch implementation
    torch_output = torch.softmax(torch.matmul(a, b), dim=-1)

    torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)

test_matmul_softmax()