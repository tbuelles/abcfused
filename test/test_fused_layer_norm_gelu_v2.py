import torch
from abcfused.ops import fused_layer_norm_gelu

def test_layer_norm_gelu():
    torch.manual_seed(0)
    N_ROWS = 128
    N_COLS = 768
    x = torch.randn(N_ROWS, N_COLS, device="cuda", dtype=torch.float32)
    weight = torch.randn(N_COLS, device="cuda", dtype=torch.float32)
    bias = torch.randn(N_COLS, device="cuda", dtype=torch.float32)
    eps = 1e-5

    # Triton version
    output_triton, mean_triton, var_triton = fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch version
    layer_norm = torch.nn.LayerNorm(N_COLS, eps=eps).to("cuda")
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    output_torch = layer_norm(x)
    output_torch = output_torch * 0.5 * (1.0 + torch.tanh(0.7978845608028654 * (output_torch + 0.044715 * output_torch * output_torch * output_torch)))
    mean_torch = torch.mean(x, dim=1)
    var_torch = torch.var(x, dim=1, unbiased=False)

    # Compare
    torch.allclose(output_triton, output_torch, rtol=1e-2, atol=1e-2)