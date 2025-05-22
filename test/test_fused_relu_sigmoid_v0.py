import torch
from abcfused.ops import fused_relu_sigmoid

def relu_sigmoid_torch(x):
    return torch.sigmoid(torch.relu(x))

def test_relu_sigmoid():
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32, requires_grad=False).cuda()
    output_torch = relu_sigmoid_torch(x)
    output_triton = fused_relu_sigmoid(x)
    assert torch.allclose(output_torch, output_triton, rtol=1e-03, atol=1e-04)
    print("✅ fused_relu_sigmoid PASSED")

test_relu_sigmoid()