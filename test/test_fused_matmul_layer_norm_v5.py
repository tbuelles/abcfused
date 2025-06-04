import torch
import fused_matmul_layer_norm.ops
import torch.nn.functional as F

torch.manual_seed(0)

def test_fused_matmul_layer_norm():
    M, K, N = 256, 128, 64
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    weight = torch.randn((N,), device='cuda', dtype=torch.float32)
    bias = torch.randn((N,), device='cuda', dtype=torch.float32)
    eps = 1e-5

    # reference
    c_ref = torch.matmul(a, b)
    mean_ref = torch.mean(c_ref, dim=1, keepdim=True)
    var_ref = torch.var(c_ref, dim=1, keepdim=True, unbiased=False)
    c_ref_norm = (c_ref - mean_ref) / torch.sqrt(var_ref + eps)
    c_ref_norm = c_ref_norm * weight + bias

    # triton
    c, mean, var = fused_matmul_layer_norm.ops.fused_matmul_layer_norm(a, b, weight, bias, eps=eps)

    assert torch.allclose(c, c_ref_norm, atol=1e-2, rtol=1e-2)
    assert torch.allclose(mean, torch.mean(c_ref, dim=1), atol=1e-2, rtol=1e-2)
    assert torch.allclose(var, torch.var(c_ref, dim=1, unbiased=False), atol=1e-2, rtol=1e-2)

test_fused_matmul_layer_norm()