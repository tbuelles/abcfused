import torch
import abcfused.ops
from abcfused.ops import fused_relu_add

def test_relu_add():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32, device='cuda')
    y = torch.randn(1024, dtype=torch.float32, device='cuda')

    # triton implementation
    output_triton = fused_relu_add(x, y)

    # reference implementation
    output_torch = torch.relu(x) + y

    # compare
    torch.testing.assert_close(output_torch, output_triton, rtol=1e-3, atol=1e-3)
    print("✅ passed!")

if __name__ == '__main__':
    test_relu_add()