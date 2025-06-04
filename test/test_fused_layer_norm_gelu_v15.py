import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

torch.manual_seed(0)

def test_fused_layer_norm_gelu():
    N = 256
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton fused kernel
    y_triton, mean_triton, rvar_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)

    # PyTorch equivalent
    x_norm = F.layer_norm(x, x.shape, weight, bias, eps)
    y_torch = x_norm * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    assert torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)

    # Calculate the mean and variance manually
    mean_torch = torch.mean(x)
    var_torch = torch.var(x, unbiased=False)
    rvar_torch = 1 / torch.sqrt(var_torch + eps)
    assert torch.allclose(torch.mean(mean_triton), mean_torch, atol=1e-5, rtol=1e-5)
    assert torch.allclose(torch.mean(rvar_triton), rvar_torch, atol=1e-5, rtol=1e-5)

test_fused_layer_norm_gelu()