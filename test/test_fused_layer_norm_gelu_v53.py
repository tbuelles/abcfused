import torch
import triton
import triton.language as tl
from fused_layer_norm_gelu import layer_norm_gelu

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(variance + eps)
    x_hat = x_hat * weight + bias
    gelu_out = x_hat * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))
    return gelu_out

torch.manual_seed(0)
N_ELEMENTS, D_MODEL = 1024, 768
x = torch.randn(N_ELEMENTS, D_MODEL, device='cuda', dtype=torch.float32)
weight = torch.randn(D_MODEL, device='cuda', dtype=torch.float32)
bias = torch.randn(D_MODEL, device='cuda', dtype=torch.float32)
eps = 1e-5

output_triton = layer_norm_gelu(x, weight, bias, eps=eps)
output_torch = layer_norm_gelu_ref(x, weight, bias, eps=eps)

print(f"Triton Output: {output_triton}")
print(f"Torch Output: {output_torch}")

assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)
print("✅ Triton and Torch outputs match!")