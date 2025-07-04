import torch
import abcfused.ops
import triton

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N_ROWS = 128
    N_COLS = 256
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # LayerNorm + GELU using Torch
    def torch_layer_norm_gelu(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(variance + eps)
        x_norm = x_norm * weight + bias
        return x_norm * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    torch_output = torch_layer_norm_gelu(x, weight, bias, eps)

    # LayerNorm + GELU using Triton
    triton_output = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

    # Compare the results
    torch.allclose(torch_output, triton_output, rtol=1e-3, atol=1e-3)