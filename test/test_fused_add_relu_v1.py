import torch
from abcfused.ops import fused_add_relu

def test_fused_add_relu():
    x = torch.randn(2, 128, device='cuda', dtype=torch.float32)
    y = torch.randn(2, 128, device='cuda', dtype=torch.float32)
    output_torch = torch.relu(x + y)
    output_triton = fused_add_relu(x, y)
    assert torch.allclose(output_torch, output_triton, rtol=1e-3, atol=1e-3)