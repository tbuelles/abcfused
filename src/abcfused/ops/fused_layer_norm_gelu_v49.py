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
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    N,  # number of rows in X
    M,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    # barrier: pointer to a shared memory for storing partial results
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE_M
    col_start = col_idx * BLOCK_SIZE_N

    # Load data
    x = tl.load(X + row_start * M + tl.arange(0, BLOCK_SIZE_M)[:, None] * M + col_start + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < N) & (col_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < M), other=0.0)

    # Compute mean
    sum_x = tl.sum(x, axis=1)
    _mean = tl.sum(sum_x, axis=0) / M
    tl.store(mean + row_idx, _mean)
    
    # Compute variance
    var = tl.sum((x - _mean)**2, axis=1)
    _var = tl.sum(var, axis=0) / M

    # Compute rstd
    _rstd = 1 / tl.sqrt(_var + eps)
    tl.store(rstd + row_idx, _rstd)

    # Normalize
    x_norm = (x - _mean) * _rstd

    # Scale and shift
    w = tl.load(W + col_start + tl.arange(0, BLOCK_SIZE_N), mask=col_start + tl.arange(0, BLOCK_SIZE_N) < M, other=0.0)
    b = tl.load(B + col_start + tl.arange(0, BLOCK_SIZE_N), mask=col_start + tl.arange(0, BLOCK_SIZE_N) < M, other=0.0)
    x_norm = x_norm * w + b
    
    # GELU
    gelu_out = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
    
    # Store the output
    tl.store(Y + row_start * M + tl.arange(0, BLOCK_SIZE_M)[:, None] * M + col_start + tl.arange(0, BLOCK_SIZE_N)[None, :], gelu_out, mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < N) & (col_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < M))


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    output = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device="cuda")
    rstd = torch.empty((N,), dtype=torch.float32, device="cuda")
    
    grid = (N // BLOCK_SIZE_M + (N % BLOCK_SIZE_M > 0), M // BLOCK_SIZE_N + (M % BLOCK_SIZE_N > 0))

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        rstd,
        N,
        M,
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output