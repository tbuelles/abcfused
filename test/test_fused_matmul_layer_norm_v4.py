import torch
import abcfused.ops
from abcfused.ops import fused_matmul_layer_norm

def test_matmul_layer_norm():
    torch.manual_seed(0)
    M, K, N = 128, 64, 32
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    w = torch.randn((N,), device='cuda', dtype=torch.float32)
    bias = torch.randn((N,), device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Triton version
    triton_output = fused_matmul_layer_norm(a, b, w, bias, eps)

    # PyTorch version
    c = torch.matmul(a, b)
    mean = torch.mean(c, dim=1, keepdim=True)
    variance = torch.var(c, unbiased=False, dim=1, keepdim=True)
    norm = (c - mean) / torch.sqrt(variance + eps)
    torch_output = norm * w + bias


    assert torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)
    print("matmul_layer_norm test passed!")

if __name__ == "__main__":
    test_matmul_layer_norm()