import torch
import abcfused.ops
from abcfused.ops import fused_matmul_layer_norm

torch.manual_seed(0)

def test_fused_matmul_layer_norm():
    N, K, M = 128, 64, 32
    A = torch.randn((N, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, M), device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton fused kernel
    triton_output = fused_matmul_layer_norm(A, B, bias, weight, eps)

    # PyTorch equivalent
    matmul_output = torch.matmul(A, B)
    layer_norm = torch.nn.LayerNorm(M, eps=eps, elementwise_affine=True).to('cuda')
    layer_norm.bias.data = bias
    layer_norm.weight.data = weight
    pytorch_output = layer_norm(matmul_output)


    torch.testing.assert_close(triton_output, pytorch_output, rtol=1e-03, atol=1e-03)


test_fused_matmul_layer_norm()