import torch
import abcfused.ops
import unittest

class TestFusedMatmulLayerNorm(unittest.TestCase):
    def test_matmul_layer_norm(self):
        torch.manual_seed(0)
        M, K, N = 128, 64, 64
        A = torch.randn((M, K), device='cuda', dtype=torch.float32)
        B = torch.randn((K, N), device='cuda', dtype=torch.float32)
        W = torch.randn((N,), device='cuda', dtype=torch.float32)
        bias = torch.randn((N,), device='cuda', dtype=torch.float32)
        eps = 1e-5

        C_triton = abcfused.ops.fused_matmul_layer_norm(A, B, W, bias, eps)

        C_matmul = torch.matmul(A, B)
        mean = torch.mean(C_matmul, dim=1, keepdim=True)
        var = torch.var(C_matmul, dim=1, keepdim=True, unbiased=False)
        C_layer_norm = (C_matmul - mean) / torch.sqrt(var + eps)
        C_layer_norm = C_layer_norm * W + bias

        self.assertTrue(torch.allclose(C_triton, C_layer_norm, atol=1e-2, rtol=1e-2))

if __name__ == '__main__':
    unittest.main()