# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    X, W, B, Output,
    Mean, Rstd,
    N_ELEMENTS: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, ROW_SIZE)
    mask = cols < ROW_SIZE

    # Load row from X
    x = tl.load(X + row_idx * ROW_SIZE + cols, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / ROW_SIZE

    # Compute variance
    variance = tl.sum((x - mean) ** 2, axis=0) / ROW_SIZE

    # Compute rstd
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Store mean and rstd
    tl.store(Mean + row_idx, mean)
    tl.store(Rstd + row_idx, rstd)

    # Normalize
    x_norm = (x - mean) * rstd

    # Apply layer norm
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    x_norm = x_norm * w + b

    # Apply GELU
    gelu = 0.5 * x_norm * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Store the result
    tl.store(Output + row_idx * ROW_SIZE + cols, gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS, ROW_SIZE = x.shape
    output = torch.empty_like(x)
    mean = torch.empty((N_ELEMENTS,), dtype=torch.float32, device='cuda')
    rstd = torch.empty((N_ELEMENTS,), dtype=torch.float32, device='cuda')

    grid = (N_ELEMENTS,)

    _fused_layer_norm_gelu_kernel[grid](
        x, weight, bias, output,
        mean, rstd,
        N_ELEMENTS=N_ELEMENTS,
        ROW_SIZE=ROW_SIZE,
        eps=eps
    )

    return output