# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    Y,  # *Pointer* to the output tensor
    X,  # *Pointer* to the input tensor
    W,  # *Pointer* to the weights tensor
    B,  # *Pointer* to the bias tensor
    MEAN,  # *Pointer* to the mean tensor
    RVAR,  # *Pointer* to the rvar tensor
    N_ELEMENTS: tl.constexpr,  # Number of elements in the input tensor
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
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
    mean = tl.sum(x, axis=1, keepdims=True) / COLS
    variance = tl.sum((x - mean) ** 2, axis=1, keepdims=True) / COLS
    x_norm = (x - mean) / tl.sqrt(variance + EPS)
    w = tl.load(W + offsets_cols[None, :], mask=offsets_cols[None, :] < COLS, other=1.0)
    b = tl.load(B + offsets_cols[None, :], mask=offsets_cols[None, :] < COLS, other=0.0)
    x_norm = x_norm * w + b

    tl.store(MEAN + offsets_rows, tl.sum(mean, axis=1), mask = offsets_rows < ROWS)
    tl.store(RVAR + offsets_rows, tl.sum(variance, axis=1), mask = offsets_rows < ROWS)


    # GELU
    gelu_out = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    tl.store(Y + offsets_rows[:, None] * COLS + offsets_cols[None, :], gelu_out, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.numel()
    ROWS, COLS = x.shape
    output = torch.empty_like(x)
    mean = torch.empty((ROWS,), device=x.device, dtype=x.dtype)
    rvar = torch.empty((ROWS,), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_COLS = 64
    BLOCK_SIZE_ROWS = 64

    grid = (triton.cdiv(ROWS, BLOCK_SIZE_ROWS), triton.cdiv(COLS, BLOCK_SIZE_COLS))

    _kernel[grid](
        output,
        x,
        weight,
        bias,
        mean,
        rvar,
        N_ELEMENTS=N_ELEMENTS,
        ROWS=ROWS,
        COLS=COLS,
        EPS=eps,
        BLOCK_SIZE_COLS=BLOCK_SIZE_COLS,
        BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
    )
    return output, mean, rvar