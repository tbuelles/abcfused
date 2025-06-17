import torch
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N, M = 256, 1024
    x = torch.randn(N, M, device=device, dtype=torch.float32)
    w = torch.randn(M, device=device, dtype=torch.float32)
    b = torch.randn(M, device=device, dtype=torch.float32)
    eps = 1e-5

    # PyTorch implementation
    def torch_layer_norm_gelu(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        x_hat = x_hat * weight + bias
        return 0.5 * x_hat * (1 + torch.tanh(0.7978845608028654 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))

    torch_output = torch_layer_norm_gelu(x, w, b, eps)

    # Triton implementation
    triton_output = fused_layer_norm_gelu.layer_norm_gelu(x, w, b, eps)

    # Compare
    torch.allclose(torch_output, triton_output, rtol=1e-3, atol=1e-3)
    assert torch.allclose(torch_output, triton_output, rtol=1e-3, atol=1e-3)

test_layer_norm_gelu()