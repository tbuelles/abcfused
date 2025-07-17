import torch
import torch.nn as nn
from abcfused.ops import fused_layer_norm_gelu

class LayerNormGELU(nn.Module):
    def __init__(self, n_features, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_features, eps=eps)
        self.eps = eps

    def forward(self, x):
        x = self.layer_norm(x)
        return torch.nn.functional.gelu(x)

if __name__ == "__main__":
    N_ELEMENTS = 128
    N_FEATURES = 768
    x = torch.randn(N_ELEMENTS, N_FEATURES, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_FEATURES, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_FEATURES, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton version
    y_triton = fused_layer_norm_gelu(x, weight, bias, eps=eps)

    # PyTorch version
    layer_norm = nn.LayerNorm(N_FEATURES, eps=eps).to('cuda').float()
    layer_norm.weight.data = weight
    layer_norm.bias.data = bias
    y_pytorch = torch.nn.functional.gelu(layer_norm(x))

    # Compare
    torch.allclose(y_triton, y_pytorch, atol=1e-2, rtol=1e-2)