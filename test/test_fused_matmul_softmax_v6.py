import torch
import abcfused.ops
from torch.testing import assert_close

def test_fused_matmul_softmax():
    M, K, N = 128, 64, 32
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # Triton
    triton_output = abcfused.ops.fused_matmul_softmax(a, b)

    # PyTorch
    torch_output = torch.matmul(a, b)
    torch_output = torch.softmax(torch_output, dim=-1)

    assert_close(triton_output, torch_output, rtol=1e-3, atol=1e-3)