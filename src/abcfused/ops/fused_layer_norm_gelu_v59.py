# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    Y,  # pointer to the output tensor
    X,  # pointer to the input tensor
    W,  # pointer to the weight tensor
    B,  # pointer to the bias tensor
    NORM_VAR, # pointer to the variance
    NORM_MEAN, # pointer to the mean
    N_ROWS,  # number of rows in the input tensor
    N_COLS,  # number of columns in the input tensor
    eps,      # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    x_ptrs = X + row_idx * N_COLS + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.)

    # Calculate mean and variance
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS

    tl.store(NORM_MEAN + row_idx, mean)
    tl.store(NORM_VAR + row_idx, variance)

    # Layer normalization
    x_norm = (x - mean) / tl.sqrt(variance + eps)
    w_ptrs = W + col_offsets
    b_ptrs = B + col_offsets
    w = tl.load(w_ptrs, mask=mask, other=1.)
    b = tl.load(b_ptrs, mask=mask, other=0.)
    x_norm = x_norm * w + b

    # GELU activation
    gelu_output = 0.5 * x_norm * (1 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    y_ptrs = Y + row_idx * N_COLS + col_offsets
    tl.store(y_ptrs, gelu_output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    output = torch.empty_like(x)
    norm_var = torch.empty((N_ROWS,), dtype=torch.float32, device=x.device)
    norm_mean = torch.empty((N_ROWS,), dtype=torch.float32, device=x.device)

    grid = (N_ROWS,)
    BLOCK_SIZE = 1024  # Adjust this based on your hardware
    _layer_norm_gelu_kernel[grid](
        output,
        x,
        weight,
        bias,
        norm_var,
        norm_mean,
        N_ROWS,
        N_COLS,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output