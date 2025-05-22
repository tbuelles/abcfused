import torch
import abcfused.ops
import torch.nn.functional as F

torch.manual_seed(42)

def test_layer_norm_gelu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Generate input data
    rows, cols = 64, 256
    x = torch.randn(rows, cols, device=device, dtype=torch.float32)
    weight = torch.randn(cols, device=device, dtype=torch.float32)
    bias = torch.randn(cols, device=device, dtype=torch.float32)
    eps = 1e-5

    # Triton output
    triton_output = abcfused.ops.fused_layer_norm_gelu(x, weight, bias, eps)

    # PyTorch equivalent
    layer_norm = torch.nn.LayerNorm(cols, eps=eps).to(device)
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    pytorch_output = F.gelu(layer_norm(x))

    # Compare
    torch.allclose(triton_output, pytorch_output, rtol=1e-3, atol=1e-3)

test_layer_norm_gelu()