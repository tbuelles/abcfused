import torch
from abcfused.ops import fused_relu_add

def test_fused_relu_add():
    torch.manual_seed(0)
    n_elements = 1024 * 10
    x = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    y = torch.randn(n_elements, device='cuda', dtype=torch.float32)

    output_triton = fused_relu_add(x, y)
    output_torch = torch.relu(x) + y

    assert torch.allclose(output_triton, output_torch), f"Triton output: {output_triton}, Torch output: {output_torch}"

if __name__ == "__main__":
    test_fused_relu_add()