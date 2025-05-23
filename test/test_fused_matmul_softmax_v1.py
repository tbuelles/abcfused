import torch
from abcfused.ops import fused_matmul_softmax

def test_fused_matmul_softmax():
    M, K, N = 64, 32, 64
    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')

    # Triton version
    triton_output = fused_matmul_softmax(a, b)

    # PyTorch version
    torch_output = torch.matmul(a, b)
    torch_output = torch.softmax(torch_output, dim=1)

    torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)
    print("Matmul + Softmax Fusion Test Passed!")

test_fused_matmul_softmax()