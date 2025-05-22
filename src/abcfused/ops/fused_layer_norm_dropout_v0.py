# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_layer_norm_dropout(
    X,  # pointer to the input
    Y,  # pointer to the output
    scale,  # pointer to the scale factor
    bias,  # pointer to the bias factor
    R,  # pointer to random numbers
    N_ROWS,  # number of rows in X
    N_COLS,  # number of columns in X
    eps,  # a small number to prevent division by zero
    p,  # dropout probability
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE_ROWS
    col_start = col_idx * BLOCK_SIZE_COLS

    offsets = tl.arange(0, BLOCK_SIZE_COLS)
    mask = offsets < N_COLS

    x_ptrs = X + row_start * N_COLS + offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Layer Norm
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    inv_std = 1.0 / tl.sqrt(variance + eps)

    scale_val = tl.load(scale + offsets, mask=mask, other=1.0)
    bias_val = tl.load(bias + offsets, mask=mask, other=0.0)

    normed_x = (x - mean) * inv_std * scale_val + bias_val

    # Dropout
    random_number = tl.rand(row_idx, col_start)
    keep_mask = random_number >= p
    output = tl.where(keep_mask, normed_x / (1 - p), 0.0)
    
    y_ptrs = Y + row_start * N_COLS + offsets
    tl.store(y_ptrs, output, mask=mask)

def fused_layer_norm_dropout(
    x: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    p: float,
    num_threads=256,
    block_size_rows=32,
    block_size_cols=32
):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    grid = (n_rows // block_size_rows + (n_rows % block_size_rows > 0), 
            n_cols // block_size_cols + (n_cols % block_size_cols > 0))
    
    _kernel_layer_norm_dropout[grid](
        x,
        output,
        scale,
        bias,
        torch.empty((1024,), device='cuda'),
        n_rows,
        n_cols,
        eps,
        p,
        BLOCK_SIZE_ROWS=block_size_rows,
        BLOCK_SIZE_COLS=block_size_cols,
        num_threads=num_threads,
    )
    return output