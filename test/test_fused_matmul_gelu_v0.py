import torch
from abcfused.ops import fused_matmul_gelu

def reference_matmul_gelu(a, b):
    x = torch.matmul(a, b)
    gelu_approx = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    return gelu_approx

torch.manual_seed(0)
a = torch.randn((128, 128), device='cuda', dtype=torch.float32)
b = torch.randn((128, 128), device='cuda', dtype=torch.float32)

triton_output = fused_matmul_gelu(a, b)
torch_output = reference_matmul_gelu(a, b)

torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)