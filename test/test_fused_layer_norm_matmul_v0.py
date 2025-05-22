import torch
from abcfused.ops import fused_layer_norm_matmul

torch.manual_seed(42)

def test_fused_layer_norm_matmul():
    ROWS = 128
    COLS = 64
    K = 32
    eps = 1e-5

    X = torch.randn(ROWS, COLS, dtype=torch.float32, device='cuda')
    W = torch.randn(COLS, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, dtype=torch.float32, device='cuda')
    Gamma = torch.randn(ROWS, dtype=torch.float32, device='cuda')
    Beta = torch.randn(ROWS, dtype=torch.float32, device='cuda')

    # Triton kernel
    triton_output = fused_layer_norm_matmul(X, W, B, Gamma, Beta, eps)

    # PyTorch equivalent
    X_mean = torch.mean(X, dim=1, keepdim=True)
    X_var = torch.var(X, dim=1, keepdim=True)
    X_norm = (X - X_mean) / torch.sqrt(X_var + eps)
    X_norm = X_norm * Gamma[:, None] + Beta[:, None]
    torch_output = torch.matmul(X_norm, W) + B

    # Assert close
    torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)

test_fused_layer_norm_matmul()