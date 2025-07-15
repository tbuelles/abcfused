import torch
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    torch.manual_seed(42)
    N_ELEMENTS = 512
    N_FEATURES = 1024
    x = torch.randn(N_ELEMENTS, N_FEATURES, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_FEATURES, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_FEATURES, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton fused kernel
    output_triton = fused_layer_norm_gelu.layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(N_FEATURES, eps=eps).to('cuda')
    layer_norm.weight = torch.nn.Parameter(weight)
    layer_norm.bias = torch.nn.Parameter(bias)
    output_torch = layer_norm(x)
    output_torch = 0.5 * output_torch * (1 + torch.tanh(0.7978845608028654 * (output_torch + 0.044715 * output_torch * output_torch * output_torch)))


    torch.testing.assert_close(output_triton, output_torch, rtol=1e-3, atol=1e-3)