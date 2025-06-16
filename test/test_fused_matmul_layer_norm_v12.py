import torch
import abcfused.ops
import torch.nn.functional as F

torch.manual_seed(0)

M, K, N = 64, 32, 16
A = torch.randn((M, K), device="cuda", dtype=torch.float32)
B = torch.randn((K, N), device="cuda", dtype=torch.float32)
W = torch.randn((M,), device="cuda", dtype=torch.float32)
B_LN = torch.randn((M,), device="cuda", dtype=torch.float32)
eps = 1e-5

C_triton, mean_triton, var_triton = abcfused.ops.fused_matmul_layer_norm(A, B, W, B_LN, eps)

C_torch = torch.matmul(A, B)
mean_torch = torch.mean(C_torch, dim = 1, keepdim = True)
var_torch = torch.var(C_torch, dim = 1, unbiased = False, keepdim = True)
C_torch_norm = (C_torch - mean_torch) / torch.sqrt(var_torch + eps)
C_torch_normed = C_torch_norm * W[:, None] + B_LN[:, None]
mean_torch = torch.mean(C_torch, dim = 1)
var_torch = torch.var(C_torch, dim = 1, unbiased = False)


torch.allclose(C_triton, C_torch_normed), torch.allclose(mean_triton, mean_torch), torch.allclose(var_triton, var_torch)