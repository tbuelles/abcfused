import torch
import abcfused.ops
import triton

def test_fused_layer_norm_gelu():
    N = 1024
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    weight = torch.randn(N, dtype=torch.float32, device='cuda')
    bias = torch.randn(N, dtype=torch.float32, device='cuda')
    eps = 1e-5

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(x.shape[-1], eps=eps).to("cuda").float()
    layer_norm.weight = torch.nn.Parameter(weight)
    layer_norm.bias = torch.nn.Parameter(bias)
    x_norm = layer_norm(x)
    gelu_ref = torch.nn.functional.gelu(x_norm)

    # Triton kernel
    gelu_triton = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps=eps)

    # Compare
    torch.allclose(gelu_triton, gelu_ref, rtol=1e-2, atol=1e-2)
    if not torch.allclose(gelu_triton, gelu_ref, rtol=1e-2, atol=1e-2):
        print("Triton Output", gelu_triton)
        print("Pytorch Output", gelu_ref)
    assert torch.allclose(gelu_triton, gelu_ref, rtol=1e-2, atol=1e-2)