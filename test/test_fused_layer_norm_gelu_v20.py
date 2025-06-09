import torch
import abcfused.ops
from abcfused.ops import fused_layer_norm_gelu

def layer_norm_gelu_ref(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=1, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(variance + eps)
    output = x_hat * weight + bias
    return output * 0.5 * (1.0 + torch.erf(output / 1.41421356237))


if __name__ == '__main__':
    torch.manual_seed(0)
    N_ROWS, N_COLS = 2048, 768
    x = torch.randn(N_ROWS, N_COLS, device='cuda', dtype=torch.float32)
    weight = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    bias = torch.randn(N_COLS, device='cuda', dtype=torch.float32)
    # triton_output = fused_layer_norm(x, weight, bias).cpu()
    triton_output = fused_layer_norm_gelu(x, weight, bias)
    torch_output = layer_norm_gelu_ref(x, weight, bias, eps=1e-5)
    print(torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3))