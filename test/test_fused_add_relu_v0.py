import torch
import abcfused.ops

def test_fused_add_relu():
    torch.manual_seed(0)
    size = 1024 * 10
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)

    output_triton = abcfused.ops.fused_add_relu(x, y)
    output_torch = torch.relu(x + y)

    torch.allclose(output_triton, output_torch)
    assert torch.allclose(output_triton, output_torch)