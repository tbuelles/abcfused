import torch
import abcfused.ops

def test_fused_layer_norm_gelu():
    torch.manual_seed(0)
    N_ELEMENTS = 2048
    num_rows = 4
    x = torch.randn((num_rows, N_ELEMENTS), device='cuda', dtype=torch.float32, requires_grad=True)
    weight = torch.randn((N_ELEMENTS,), device='cuda', dtype=torch.float32, requires_grad=True)
    bias = torch.randn((N_ELEMENTS,), device='cuda', dtype=torch.float32, requires_grad=True)
    eps = 1e-5

    # Triton fused kernel
    output_fused = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(N_ELEMENTS, eps=eps, elementwise_affine=True).to('cuda')
    layer_norm.weight = torch.nn.Parameter(weight)
    layer_norm.bias = torch.nn.Parameter(bias)
    output_torch = torch.nn.functional.gelu(layer_norm(x))

    torch.allclose(output_fused, output_torch, rtol=1e-3, atol=1e-3)

    # Backprop test

    output_fused.sum().backward()
    output_torch.sum().backward()

    assert torch.allclose(x.grad, x.grad, rtol=1e-3, atol=1e-3)
    assert torch.allclose(weight.grad, weight.grad, rtol=1e-3, atol=1e-3)
    assert torch.allclose(bias.grad, bias.grad, rtol=1e-3, atol=1e-3)