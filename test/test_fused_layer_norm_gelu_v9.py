import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N_ROWS = 128
    N_COLS = 768
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # triton output
    output_triton = fused_layer_norm_gelu.layer_norm_gelu(x, weight, bias, eps)

    # pytorch output
    x_mean = torch.mean(x, dim=1, keepdim=True)
    x_var = torch.var(x, dim=1, keepdim=True, unbiased=False)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    output_torch = x_hat * weight + bias
    output_torch = output_torch * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (output_torch + 0.044715 * output_torch * output_torch * output_torch)))

    # compare
    torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)
    assert torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)