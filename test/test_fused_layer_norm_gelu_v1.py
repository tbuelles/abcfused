import torch
import torch.nn as nn
from abcfused.ops import fused_layer_norm_gelu

class LayerNormGELU(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight * x + self.bias
        return x * 0.5 * (1.0 + torch.tanh(0.7978845608 * x * (1 + 0.044715 * x * x)))

if __name__ == '__main__':
    torch.manual_seed(42)
    N, M = 64, 256
    x = torch.randn(N, M, device='cuda', dtype=torch.float32)
    weight = torch.randn(M, device='cuda', dtype=torch.float32)
    bias = torch.randn(M, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton
    output_triton, mean_triton, var_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)

    # PyTorch
    layer_norm_gelu = LayerNormGELU(M, eps=eps).to('cuda').float()
    layer_norm_gelu.weight.data = weight
    layer_norm_gelu.bias.data = bias
    output_torch = layer_norm_gelu(x)

    assert torch.allclose(output_triton, output_torch, atol=1e-3, rtol=1e-3)
    print("✅ Triton and PyTorch match!")