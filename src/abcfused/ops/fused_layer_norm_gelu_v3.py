# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the bias
    NORM_MEAN, # pointer to the intermediate mean
    NORM_VAR,  # pointer to the intermediate variance
    N_ELEMENTS,  # number of elements in the input
    ROWS, # number of rows in the input
    COLS, # number of columns in the input
    eps,  # epsilon to avoid division by zero
    # Meta-parameters
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    row_start = row_idx * BLOCK_SIZE_ROWS
    col_start = col_idx * BLOCK_SIZE_COLS

    offsets_rows = row_start + tl.arange(0, BLOCK_SIZE_ROWS)
    offsets_cols = col_start + tl.arange(0, BLOCK_SIZE_COLS)

    mask = (offsets_rows[:, None] < ROWS) & (offsets_cols[None, :] < COLS)

    x = tl.load(X + offsets_rows[:, None] * COLS + offsets_cols[None, :], mask=mask, other=0.0)

    # LayerNorm
    mean = tl.sum(x, axis=1) / COLS
    variance = tl.sum((x - mean[:, None])**2, axis=1) / COLS

    norm_mean = tl.where(offsets_rows < ROWS, mean, 0.0)
    norm_var = tl.where(offsets_rows < ROWS, variance, 0.0)

    tl.store(NORM_MEAN + offsets_rows, norm_mean, mask= offsets_rows < ROWS)
    tl.store(NORM_VAR + offsets_rows, norm_var, mask= offsets_rows < ROWS)

    x_norm = (x - mean[:, None]) / tl.sqrt(variance[:, None] + eps)
    w = tl.load(W + offsets_cols[None, :], mask=offsets_cols[None, :] < COLS, other=1.0)
    b = tl.load(B + offsets_cols[None, :], mask=offsets_cols[None, :] < COLS, other=0.0)
    x_norm = x_norm * w[None, :] + b[None, :]

    # GELU
    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * x_norm * (1.0 + 0.044715 * x_norm * x_norm)))

    tl.store(Y + offsets_rows[:, None] * COLS + offsets_cols[None, :], gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    rows, cols = x.shape
    n_elements = rows * cols

    output = torch.empty_like(x)
    norm_mean = torch.empty((rows,), dtype=torch.float32, device=x.device)
    norm_var = torch.empty((rows,), dtype=torch.float32, device=x.device)

    grid = (rows, (cols + 128 - 1) // 128)

    _layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        norm_mean,
        norm_var,
        n_elements,
        rows,
        cols,
        eps,
        BLOCK_SIZE_ROWS=1,
        BLOCK_SIZE_COLS=128,
    )
    return output