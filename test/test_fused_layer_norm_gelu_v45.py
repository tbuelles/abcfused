import torch
import abcfused.ops

def test_layer_norm_gelu():
    N_ROWS = 128
    N_FEATURES = 256
    x = torch.randn(N_ROWS, N_FEATURES, dtype=torch.float32, device='cuda')
    weight = torch.randn(N_FEATURES, dtype=torch.float32, device='cuda')
    bias = torch.randn(N_FEATURES, dtype=torch.float32, device='cuda')
    eps = 1e-5

    # PyTorch implementation
    def layer_norm_gelu(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(variance + eps)
        x_normed = x_hat * weight + bias
        gelu_val = 0.5 * x_normed * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x_normed + 0.044715 * x_normed * x_normed * x_normed)))
        return gelu_val

    expected = layer_norm_gelu(x, weight, bias, eps)

    # Triton implementation
    actual = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

    torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)