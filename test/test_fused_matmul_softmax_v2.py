import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from abcfused.ops import fused_matmul_softmax

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float32)
b = torch.randn((512, 512), device='cuda', dtype=torch.float32)

# Triton kernel
triton_output = fused_matmul_softmax(a, b)

# PyTorch equivalent
torch_output = F.softmax(torch.matmul(a, b), dim=-1)

# Compare
torch.allclose(triton_output, torch_output, atol=1e-3, rtol=1e-3)