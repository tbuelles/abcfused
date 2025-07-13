# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    norm_eps,  # normalization epsilon
    N,  # number of rows in X
    M,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program ID to the row of X and Y it should compute.
    row_idx = tl.program_id(0)
    # X and Y have the same shape.
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    # Load data from x
    x = tl.load(X + row_idx * M + offsets, mask=mask)

    # Compute mean
    x_mean = tl.sum(x, axis=0) / M
    x_mean = tl.where(mask, x_mean, 0.) # Ensure correct mean computation with padding
    
    # Compute variance
    x_var = tl.sum((x - x_mean) * (x - x_mean), axis=0) / M
    x_var = tl.where(mask, x_var, 0.) # Ensure correct variance computation with padding

    # Normalize
    x_norm = (x - x_mean) / tl.sqrt(x_var + norm_eps)

    # Weight and bias
    w = tl.load(W + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    x_norm = x_norm * w + b

    # GELU activation
    gelu_out = x_norm * 0.5 * (1.0 + tl.erf(x_norm / 1.41421356237))

    # Write back to Y
    tl.store(Y + row_idx * M + offsets, gelu_out, mask=mask)


def layer_norm_gelu(x, weight, bias, eps):
    N, M = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = min(M, 256)
    grid = (N,)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        y,
        weight,
        bias,
        eps,
        N,
        M,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y