import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu
import torch.nn as nn

class LayerNormGELU(nn.Module):
    def __init__(self, num_features, eps=1e-5, gelu_approx=0):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.gelu_approx = gelu_approx

    def forward(self, x):
        x = self.ln(x)
        if self.gelu_approx == 0:
            return x * 0.5 * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))
        else:
            return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

if __name__ == '__main__':
    N_ROW = 128
    N_COL = 768
    x = torch.randn(N_ROW, N_COL, dtype=torch.float32, device='cuda')
    weight = torch.randn(N_COL, dtype=torch.float32, device='cuda')
    bias = torch.randn(N_COL, dtype=torch.float32, device='cuda')
    eps = 1e-5
    gelu_approx = 0

    # Triton fused kernel
    output_triton = fused_layer_norm_gelu(x, weight, bias, eps, gelu_approx)

    # PyTorch equivalent
    layer_norm_gelu = LayerNormGELU(N_COL, eps, gelu_approx).cuda()
    layer_norm_gelu.weight.data = weight
    layer_norm_gelu.bias.data = bias
    with torch.no_grad():
        output_torch = layer_norm_gelu(x)

    # Compare
    torch.set_printoptions(precision=10)
    assert torch.allclose(output_triton, output_torch, atol=1e-3, rtol=1e-3)
    print("Test passed!")