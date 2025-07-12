# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B,
    Y,
    Mean, Var,
    N_ELEMENTS: tl.constexpr,
    eps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, N_ELEMENTS)
    x = tl.load(X + row_idx * N_ELEMENTS + cols, mask=cols < N_ELEMENTS, other=0.0)

    # Calculate mean
    mean = tl.sum(x, axis=0) / N_ELEMENTS
    tl.store(Mean + row_idx, mean)

    # Calculate variance
    var = tl.sum((x - mean)**2, axis=0) / N_ELEMENTS
    tl.store(Var + row_idx, var)

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)

    # Layer Norm
    w = tl.load(W + cols, mask=cols < N_ELEMENTS, other=1.0)
    b = tl.load(B + cols, mask=cols < N_ELEMENTS, other=0.0)
    x_norm = x_norm * w + b

    # GELU activation
    gelu_val = 0.5 * x_norm * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.pi) * (x_norm + 0.044715 * x_norm**3)))

    tl.store(Y + row_idx * N_ELEMENTS + cols, gelu_val, mask=cols < N_ELEMENTS)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, N_ELEMENTS = x.shape
    output = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device='cuda')
    var = torch.empty((N,), dtype=torch.float32, device='cuda')

    grid = (N,)
    _layer_norm_gelu_kernel[grid](
        x, weight, bias,
        output,
        mean, var,
        N_ELEMENTS=N_ELEMENTS,
        eps=eps
    )
    return output