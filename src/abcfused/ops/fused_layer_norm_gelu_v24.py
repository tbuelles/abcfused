# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B, Y,  # Pointers to data
    N,  # Number of rows in X
    M,  # Number of columns in X
    eps,  # LayerNorm epsilon
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE_N
    col_start = col_idx * BLOCK_SIZE_M

    # Compute mean
    row = row_start + tl.arange(0, BLOCK_SIZE_N)
    mask = row < N
    x_ptr = X + row[:, None] * M + tl.arange(0, BLOCK_SIZE_M)
    x = tl.load(x_ptr, mask=mask[:, None], other=0.0)
    mean = tl.sum(x, axis=1) / M

    # Compute variance
    x_mean = x - mean[:, None]
    variance = tl.sum(x_mean * x_mean, axis=1) / M

    # Normalize
    inv_std = 1.0 / tl.sqrt(variance + eps)
    x_norm = x_mean * inv_std[:, None]

    # Layer norm
    w_ptr = W + tl.arange(0, BLOCK_SIZE_M)
    b_ptr = B + tl.arange(0, BLOCK_SIZE_M)
    w = tl.load(w_ptr, mask=tl.arange(0, BLOCK_SIZE_M) < M, other=1.0)
    b = tl.load(b_ptr, mask=tl.arange(0, BLOCK_SIZE_M) < M, other=0.0)
    x_norm = x_norm * w + b

    # GELU
    gelu_out = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Store output
    y_ptr = Y + row[:, None] * M + tl.arange(0, BLOCK_SIZE_M)
    tl.store(y_ptr, gelu_out, mask=mask[:, None])


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)

    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 64
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))

    _layer_norm_gelu_kernel[grid](
        x, weight, bias, y,
        N, M, eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return y