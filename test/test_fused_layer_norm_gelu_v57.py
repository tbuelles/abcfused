import torch
from abcfused.ops import fused_layer_norm_gelu
import torch.nn.functional as F

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    x_norm = F.layer_norm(x, (x.size(-1),), weight, bias, eps=eps)
    return F.gelu(x_norm)

torch.manual_seed(0)
N, M = 16, 2048
x = torch.randn(N, M, dtype=torch.float32, requires_grad=False).cuda()
weight = torch.randn(M, dtype=torch.float32, requires_grad=False).cuda()
bias = torch.randn(M, dtype=torch.float32, requires_grad=False).cuda()
y_triton = fused_layer_norm_gelu(x, weight, bias)
y_torch = layer_norm_gelu_ref(x, weight, bias)
assert torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)