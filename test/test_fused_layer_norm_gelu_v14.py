import torch
import abcfused.ops

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N_ROWS = 128
    N_COLS = 256
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton implementation
    y_triton = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps=eps)

    # PyTorch implementation
    layer_norm = torch.nn.LayerNorm(N_COLS, eps=eps).to('cuda')
    layer_norm.weight = torch.nn.Parameter(weight)
    layer_norm.bias = torch.nn.Parameter(bias)
    y_torch = layer_norm(x)
    y_torch = y_torch * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (y_torch + 0.044715 * y_torch * y_torch * y_torch)))

    torch.testing.assert_close(y_triton, y_torch, rtol=1e-3, atol=1e-3)


test_layer_norm_gelu()