import torch
from abcfused.ops import fused_matmul_softmax
import torch.nn.functional as F

torch.manual_seed(0)

def test_fused_matmul_softmax():
    M, K, N = 64, 32, 128
    a = torch.randn((M, K), dtype=torch.float32, device='cuda', requires_grad=False)
    b = torch.randn((K, N), dtype=torch.float32, device='cuda', requires_grad=False)

    # Triton implementation
    c_triton = fused_matmul_softmax(a, b)

    # PyTorch equivalent
    c_torch = torch.matmul(a, b)
    c_torch = F.softmax(c_torch, dim=-1)

    # Compare
    torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2)
    assert torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2)

test_fused_matmul_softmax()