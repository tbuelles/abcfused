import torch
import abcfused.ops

def test_fused_matmul_layer_norm():
    M = 64
    N = 64
    K = 64
    eps = 1e-5
    
    A = torch.randn((M, K), dtype=torch.float32, device='cuda')
    B = torch.randn((K, N), dtype=torch.float32, device='cuda')
    W = torch.randn((N,), dtype=torch.float32, device='cuda')
    B_ln = torch.randn((N,), dtype=torch.float32, device='cuda')
    
    # Triton
    C_triton = abcfused.ops.fused_matmul_layer_norm(A, B, W, B_ln, eps)
    
    # PyTorch
    C_torch = torch.matmul(A, B)
    mean = torch.mean(C_torch, dim=1, keepdim=True)
    variance = torch.var(C_torch, dim=1, keepdim=True)
    C_normalized = (C_torch - mean) / torch.sqrt(variance + eps)
    C_torch = C_normalized * W + B_ln

    torch.testing.assert_close(C_triton, C_torch, rtol=1e-3, atol=1e-3)