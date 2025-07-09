import torch
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N, M = 64, 512
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    w = torch.randn(M, device='cuda', dtype=torch.float32)
    b = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton
    y_triton = fused_layer_norm_gelu.layer_norm_gelu(x, w, b, eps)

    # PyTorch
    layer_norm = torch.nn.LayerNorm(M, eps=eps).to('cuda').to(torch.float32)
    layer_norm.weight = torch.nn.Parameter(w)
    layer_norm.bias = torch.nn.Parameter(b)
    y_torch_ln = layer_norm(x)
    y_torch = 0.5 * y_torch_ln * (1.0 + torch.tanh(0.7978845608028654 * (y_torch_ln + 0.044715 * y_torch_ln * y_torch_ln * y_torch_ln)))

    torch.testing.assert_close(y_triton, y_torch, rtol=1e-3, atol=1e-3)

test_layer_norm_gelu()