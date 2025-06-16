import torch
import abcfused.ops

def test_layer_norm_gelu():
    torch.manual_seed(42)
    N, M = 2, 1024
    x = torch.randn(N, M, dtype=torch.float32, device='cuda')
    weight = torch.randn(M, dtype=torch.float32, device='cuda')
    bias = torch.randn(M, dtype=torch.float32, device='cuda')
    eps = 1e-5

    # Reference PyTorch implementation
    def ref_layer_norm_gelu(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        y = x_hat * weight + bias
        gelu_val = 0.5 * y * (1.0 + torch.tanh(0.7978845608028654 * (y + 0.044715 * y * y * y)))
        return gelu_val

    # Run Triton kernel
    y_triton = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

    # Run PyTorch reference
    y_torch = ref_layer_norm_gelu(x, weight, bias, eps)

    # Compare
    torch.set_printoptions(precision=10)
    assert torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)
    print("Test passed!")

test_layer_norm_gelu()