import torch
import abcfused.ops

def test_fused_relu_add():
    shape = (1024, 1024)
    x = torch.randn(shape, dtype=torch.float32, device='cuda')
    y = torch.randn(shape, dtype=torch.float32, device='cuda')

    triton_output = abcfused.ops.fused_relu_add(x, y)
    torch_output = torch.relu(x) + y

    assert torch.allclose(triton_output, torch_output, rtol=1e-03, atol=1e-04)