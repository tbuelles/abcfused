import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N, M = 128, 256
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    weight = torch.randn(M, device='cuda', dtype=torch.float32)
    bias = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # triton implementation
    output_triton = fused_layer_norm_gelu(x, weight, bias, eps)

    # reference implementation
    layer_norm = torch.nn.LayerNorm(M, eps=eps).to('cuda').to(torch.float32)
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    output_ref = layer_norm(x)
    output_ref = torch.nn.functional.gelu(output_ref)

    torch.testing.assert_close(output_triton, output_ref, rtol=1e-3, atol=1e-3)
    print("✅ layer_norm_gelu test passed!")

test_layer_norm_gelu()