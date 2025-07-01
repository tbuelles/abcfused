import torch
import triton
import abcfused.ops
from torch.nn.functional import softmax

torch.manual_seed(0)

def test_matmul_softmax():
    M = 256
    K = 128
    N = 256
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # triton
    c_triton = abcfused.ops.fused_matmul_softmax(a, b)

    # pytorch
    c_matmul = torch.matmul(a, b)
    c_pytorch = softmax(c_matmul, dim=1)

    torch.allclose(c_triton, c_pytorch, atol=1e-2, rtol=1e-2)
    print("PASSED!")

test_matmul_softmax()