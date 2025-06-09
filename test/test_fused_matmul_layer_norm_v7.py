import torch
import abcfused.ops
from abcfused.ops import fused_matmul_layer_norm

def test_matmul_layer_norm():
    M = 256
    N = 128
    K = 64
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    W = torch.randn((M,), device='cuda', dtype=torch.float32)
    B_layer_norm = torch.randn((M,), device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton fused kernel
    C_triton = fused_matmul_layer_norm(A, B, W, B_layer_norm, eps)

    # PyTorch equivalent
    C_matmul = torch.matmul(A, B)
    mean = torch.mean(C_matmul, dim=1, keepdim=True)
    var = torch.var(C_matmul, dim=1, keepdim=True, unbiased=False)
    C_layer_norm = (C_matmul - mean) / torch.sqrt(var + eps)
    C_pytorch = C_layer_norm * W[:, None] + B_layer_norm[:, None]

    assert torch.allclose(C_triton, C_pytorch, atol=1e-3, rtol=1e-3)
    print("test_matmul_layer_norm passed!")

if __name__ == '__main__':
    test_matmul_layer_norm()