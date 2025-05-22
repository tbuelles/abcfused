import torch
from abcfused.ops import fused_relu_add

def test_fused_relu_add():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32)
    y = torch.randn(1024, dtype=torch.float32)

    output_triton = fused_relu_add(x, y)
    output_torch = torch.relu(x) + y

    assert torch.allclose(output_triton, output_torch)

test_fused_relu_add()