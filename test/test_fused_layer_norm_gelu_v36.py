import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N, M = 64, 512
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    weight = torch.randn(M, device='cuda', dtype=torch.float32)
    bias = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # PyTorch version
    layer_norm = torch.nn.LayerNorm(M, eps=eps, elementwise_affine=True).to('cuda')
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    torch_output = torch.nn.functional.gelu(layer_norm(x))

    # Triton version
    triton_output = fused_layer_norm_gelu.layer_norm_gelu(x, weight, bias, eps)

    torch.allclose(torch_output, triton_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(torch_output, triton_output, atol=1e-3, rtol=1e-3)