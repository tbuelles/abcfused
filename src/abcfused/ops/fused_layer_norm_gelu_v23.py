# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X,  # pointer to the input tensor
    Y,  # pointer to the output tensor
    W,  # pointer to the weights tensor
    B,  # pointer to the bias tensor
    N_ROWS,  # number of rows in X
    N_COLS,  # number of columns in X
    eps,  # epsilon to avoid dividing by zero
    # scalar configs
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    X_ptrs = X + row_idx * N_COLS + col_offsets
    # Load the row into registers
    x = tl.load(X_ptrs, mask=col_offsets < N_COLS, other=0.)
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    # Normalize
    x_norm = (x - mean) / tl.sqrt(variance + eps)
    # Scale and bias
    w = tl.load(W + col_offsets, mask=col_offsets < N_COLS, other=1.)
    b = tl.load(B + col_offsets, mask=col_offsets < N_COLS, other=0.)
    x_norm = x_norm * w + b
    # GELU
    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
    # Write back the output
    Y_ptrs = Y + row_idx * N_COLS + col_offsets
    tl.store(Y_ptrs, gelu, mask=col_offsets < N_COLS)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    output = torch.empty_like(x)
    grid = (N_ROWS,)
    _layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        N_ROWS,
        N_COLS,
        eps,
        BLOCK_SIZE=1024,
    )
    return output