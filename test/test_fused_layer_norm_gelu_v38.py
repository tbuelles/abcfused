import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    x_norm = F.layer_norm(x, (x.shape[-1],), weight, bias, eps=eps)
    return x_norm * 0.5 * (1.0 + torch.erf(x_norm / (2**0.5)))


torch.manual_seed(0)
N = 1024
x = torch.randn(N, device='cuda', dtype=torch.float32)
weight = torch.randn(N, device='cuda', dtype=torch.float32)
bias = torch.randn(N, device='cuda', dtype=torch.float32)

# triton
output_triton = fused_layer_norm_gelu(x, weight, bias)

# reference
output_ref = layer_norm_gelu_ref(x, weight, bias)

assert torch.allclose(output_triton, output_ref, atol=1e-3, rtol=1e-3)