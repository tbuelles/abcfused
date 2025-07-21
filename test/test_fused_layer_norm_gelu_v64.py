import torch
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(variance + eps)
    gelu = x_norm * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
    return gelu * weight + bias


def test_fused_layer_norm_gelu():
    torch.manual_seed(0)
    N_ROWS = 32
    N_COLS = 64
    x = torch.randn(N_ROWS, N_COLS, dtype=torch.float32, device='cuda')
    weight = torch.randn(N_COLS, dtype=torch.float32, device='cuda')
    bias = torch.randn(N_COLS, dtype=torch.float32, device='cuda')
    eps = 1e-5

    output_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)
    output_torch = layer_norm_gelu_ref(x, weight, bias, eps=eps)

    torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)
    assert torch.allclose(output_triton, output_torch, rtol=1e-3, atol=1e-3)