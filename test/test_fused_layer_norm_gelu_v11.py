import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn as nn

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N = 256
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton fused kernel
    output_triton = fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    layer_norm = nn.LayerNorm(x.shape[-1], elementwise_affine=True, eps=eps).cuda()
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    output_pytorch = layer_norm(x)
    output_pytorch = 0.5 * output_pytorch * (1.0 + torch.tanh(0.7978845608 * (output_pytorch + 0.044715 * output_pytorch * output_pytorch * output_pytorch)))

    assert torch.allclose(output_triton, output_pytorch, atol=1e-2, rtol=1e-2), f"Max diff: {torch.max(torch.abs(output_triton - output_pytorch))}"

test_layer_norm_gelu()