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
    NORM_VAR,
    NORM_MEAN,
    N_ROWS,  # number of rows in X
    N_COLS,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    # BLOCK_SIZE: it is assumed that N_COLS is a multiple of BLOCK_SIZE
    BLOCK_SIZE: tl.constexpr,
):
    # Map the thread program ID to the row of X and Y it should compute.
    row_idx = tl.program_id(0)
    # The stride represents how much the pointer needs to advance to access the next row.
    row_start_ptr = X + row_idx * N_COLS
    # block start ptr
    W_block_ptr = W
    B_block_ptr = B

    # load data
    x = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N_COLS, other=0.0)

    # compute mean
    mean = tl.sum(x, axis=0) / N_COLS

    # compute variance
    variance = tl.sum((x - mean)**2, axis=0) / N_COLS

    # layer norm
    x_norm = (x - mean) / tl.sqrt(variance + eps)
    w = tl.load(W_block_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N_COLS, other=1.0)
    b = tl.load(B_block_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N_COLS, other=0.0)
    x_norm = x_norm * w + b
    tl.store(NORM_MEAN + row_idx, mean)
    tl.store(NORM_VAR + row_idx, variance)

    # gelu
    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * x_norm * (1.0 + 0.044715 * x_norm * x_norm)))

    # write back to Y
    tl.store(Y + row_idx * N_COLS + tl.arange(0, BLOCK_SIZE), gelu, mask=tl.arange(0, BLOCK_SIZE) < N_COLS)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    NORM_MEAN = torch.empty((N_ROWS,), device=x.device, dtype=x.dtype)
    NORM_VAR = torch.empty((N_ROWS,), device=x.device, dtype=x.dtype)
    output = torch.empty_like(x)
    BLOCK_SIZE = min(triton.next_power_of_2(N_COLS), 2048)
    grid = (N_ROWS,)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        NORM_VAR,
        NORM_MEAN,
        N_ROWS,
        N_COLS,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output