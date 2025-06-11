import torch
import abcfused.ops

def test_fused_matmul_softmax():
    M, K, N = 128, 64, 256
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # Triton
    triton_output = abcfused.ops.fused_matmul_softmax(a, b)

    # PyTorch
    matmul_output = torch.matmul(a, b)
    max_vals = torch.max(matmul_output, dim=1, keepdim=True)[0]
    exp_values = torch.exp(matmul_output - max_vals)
    softmax_output = exp_values / torch.sum(exp_values, dim=1, keepdim=True)

    # Compare
    torch.allclose(triton_output, softmax_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(triton_output, softmax_output, atol=1e-3, rtol=1e-3)