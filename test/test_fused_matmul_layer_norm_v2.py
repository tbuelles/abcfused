import torch
import abcfused.ops

def test_fused_matmul_layer_norm():
    M, K, N = 64, 32, 64
    a = torch.randn(M, K, dtype=torch.float32, device='cuda')
    b = torch.randn(K, N, dtype=torch.float32, device='cuda')
    weight = torch.randn(M, dtype=torch.float32, device='cuda')
    bias = torch.randn(M, dtype=torch.float32, device='cuda')
    eps = 1e-5

    # Triton fused kernel
    triton_output = abcfused.ops.fused_matmul_layer_norm(a, b, weight, bias, eps)

    # PyTorch equivalent
    matmul_output = torch.matmul(a, b)
    mean = torch.mean(matmul_output, dim=1, keepdim=True)
    var = torch.var(matmul_output, dim=1, keepdim=True, unbiased=False)
    layer_norm_output = (matmul_output - mean) / torch.sqrt(var + eps)
    torch_output = layer_norm_output * weight[:, None] + bias[:, None]

    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2)