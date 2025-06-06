import torch
import abcfused.ops

def test_matmul_softmax():
    torch.manual_seed(42)
    M, K, N = 128, 64, 32

    a = torch.randn((M, K), dtype=torch.float32, device='cuda')
    b = torch.randn((K, N), dtype=torch.float32, device='cuda')

    # Triton version
    triton_output = abcfused.ops.fused_matmul_softmax(a, b)

    # PyTorch version
    torch_output = torch.matmul(a, b)
    torch_output = torch.softmax(torch_output, dim=-1)

    assert torch.allclose(triton_output, torch_output, rtol=1e-3, atol=1e-3)
    print("Test passed!")

if __name__ == "__main__":
    test_matmul_softmax()