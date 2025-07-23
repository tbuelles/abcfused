# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    X,  # pointer to the input tensor
    Y,  # pointer to the output tensor
    W,  # pointer to the weight tensor (LayerNorm)
    B,  # pointer to the bias tensor (LayerNorm)
    N_ROWS,  # number of rows in X
    N_COLS,  # number of columns in X
    eps,  # epsilon for LayerNorm
    # Meta-programming parameters
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE_ROWS
    col_start = col_idx * BLOCK_SIZE_COLS

    row_mask = (row_start + tl.arange(0, BLOCK_SIZE_ROWS)) < N_ROWS
    col_mask = (col_start + tl.arange(0, BLOCK_SIZE_COLS)) < N_COLS

    x = tl.load(X + row_start * N_COLS + (col_start + tl.arange(0, BLOCK_SIZE_COLS))[:, None], mask=col_mask, other=0.0)

    # LayerNorm
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    x_norm = (x - mean) / tl.sqrt(variance + eps)
    x_norm = x_norm * tl.load(W + (col_start + tl.arange(0, BLOCK_SIZE_COLS))[:, None], mask=col_mask, other=1.0) + tl.load(B + (col_start + tl.arange(0, BLOCK_SIZE_COLS))[:, None], mask=col_mask, other=0.0)

    # GELU
    sqrt_2_over_pi = 0.7978845608
    gelu = 0.5 * x_norm * (1 + tl.tanh(sqrt_2_over_pi * (x_norm + 0.044715 * x_norm ** 3)))

    tl.store(Y + row_start * N_COLS + (col_start + tl.arange(0, BLOCK_SIZE_COLS))[:, None], gelu, mask=col_mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE_ROWS = 1
    BLOCK_SIZE_COLS = 256

    grid = (n_rows // BLOCK_SIZE_ROWS, (n_cols + BLOCK_SIZE_COLS - 1) // BLOCK_SIZE_COLS)

    _kernel[grid](
        x,
        output,
        weight,
        bias,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
        BLOCK_SIZE_COLS=BLOCK_SIZE_COLS
    )

    return output