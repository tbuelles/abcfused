import torch
import abcfused.ops
from abcfused.ops import fused_matmul_layer_norm

torch.manual_seed(42)

def test_fused_matmul_layer_norm():
    M, K, N = 64, 32, 64
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    W = torch.randn((N,), device='cuda', dtype=torch.float32)
    B_layernorm = torch.randn((N,), device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton
    C_triton, _, _ = fused_matmul_layer_norm(A, B, W, B_layernorm, eps)

    # PyTorch
    C_matmul = torch.matmul(A, B)
    mean = torch.mean(C_matmul, dim=1, keepdim=True)
    variance = torch.var(C_matmul, dim=1, keepdim=True, unbiased=False)
    C_layernorm = (C_matmul - mean) / torch.sqrt(variance + eps)
    C_pytorch = C_layernorm * W + B_layernorm

    torch.testing.assert_close(C_triton, C_pytorch, rtol=1e-03, atol=1e-03)
    print("Test passed!")

if __name__ == "__main__":
    test_fused_matmul_layer_norm()