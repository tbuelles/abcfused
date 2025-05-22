import torch
import abcfused.ops
from abcfused.ops import fused_add_relu


def test_fused_add_relu():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32, device="cuda")
    y = torch.randn(1024, dtype=torch.float32, device="cuda")

    output_triton = fused_add_relu(x, y)
    output_torch = torch.relu(x + y)

    torch.allclose(output_triton, output_torch, rtol=1e-05, atol=1e-08)