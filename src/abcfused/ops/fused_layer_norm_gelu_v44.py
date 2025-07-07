# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B,
    Y,
    N,  # Number of rows in X
    M,  # Number of columns in X
    eps,  # layer norm epsilon
    # Other params
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    # Load row
    x = tl.load(X + row_idx * M + offsets, mask=mask, other=0.0)

    # Compute mean
    x_mean = tl.sum(x, axis=0) / M

    # Compute variance
    x_var = tl.sum((x - x_mean) * (x - x_mean), axis=0) / M

    # Normalize
    x_std = tl.sqrt(x_var + eps)
    x_norm = (x - x_mean) / x_std

    # Scale and bias
    x_scaled = x_norm * tl.load(W + offsets, mask=mask, other=1.0) + tl.load(B + offsets, mask=mask, other=0.0)

    # Gelu
    gelu_val = 0.5 * x_scaled * (1.0 + tl.tanh(0.7978845608028654 * (x_scaled + 0.044715 * x_scaled * x_scaled * x_scaled)))

    # Store output
    tl.store(Y + row_idx * M + offsets, gelu_val, mask=mask)

def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(M)
    grid = (N,)

    _layer_norm_gelu_kernel[grid](
        x, weight, bias,
        y,
        N, M, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return y