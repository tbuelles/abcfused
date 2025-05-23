import torch
import abcfused.ops
import torch.nn.functional as F

def test_layer_norm_gelu():
    torch.manual_seed(42)
    N, D = 32, 64
    x = torch.randn(N, D, dtype=torch.float32, requires_grad=True, device="cuda")
    weight = torch.randn(D, dtype=torch.float32, requires_grad=True, device="cuda")
    bias = torch.randn(D, dtype=torch.float32, requires_grad=True, device="cuda")
    eps = 1e-5

    # PyTorch implementation
    def torch_layer_norm_gelu(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(variance + eps)
        x_norm = x_norm * weight + bias
        return F.gelu(x_norm)

    torch_output = torch_layer_norm_gelu(x, weight, bias, eps)

    # Triton implementation
    triton_output = abcfused.ops.layer_norm_gelu(x, weight, bias, eps)

    # Compare the results
    assert torch.allclose(torch_output, triton_output, rtol=1e-3, atol=1e-3)
    print("Test passed!")

test_layer_norm_gelu()