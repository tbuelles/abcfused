import torch
from abcfused.ops import fused_layer_norm_matmul

def test_fused_layer_norm_matmul():
    M, K, N = 64, 32, 16
    x = torch.randn(M, K, dtype=torch.float32, device='cuda')
    W = torch.randn(K, N, dtype=torch.float32, device='cuda')
    weight = torch.randn(K, dtype=torch.float32, device='cuda')
    bias = torch.randn(K, dtype=torch.float32, device='cuda')

    # Triton
    output_triton = fused_layer_norm_matmul(x, W, bias, weight)

    # PyTorch
    layer_norm = torch.nn.LayerNorm(K, elementwise_affine=True).to('cuda')
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    x_norm = layer_norm(x)
    output_torch = torch.matmul(x_norm, W)

    torch.testing.assert_close(output_triton, output_torch, rtol=1e-03, atol=1e-03)

test_fused_layer_norm_matmul()