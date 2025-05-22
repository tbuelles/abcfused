import torch
import abcfused.ops
from abcfused.ops import fused_matmul_softmax

torch.manual_seed(0)

def test_fused_matmul_softmax():
    M, K, N = 64, 32, 64
    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')

    # Triton
    triton_output = fused_matmul_softmax(a, b)

    # PyTorch
    matmul_output = torch.matmul(a, b)
    max_vals = torch.max(matmul_output, dim=1, keepdim=True)[0]
    exp_values = torch.exp(matmul_output - max_vals)
    pytorch_output = exp_values / torch.sum(exp_values, dim=1, keepdim=True)

    torch.allclose(triton_output, pytorch_output, rtol=1e-03, atol=1e-03)
    assert torch.allclose(triton_output, pytorch_output, rtol=1e-03, atol=1e-03)

test_fused_matmul_softmax()