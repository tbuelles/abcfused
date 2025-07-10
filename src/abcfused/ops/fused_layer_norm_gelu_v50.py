# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    NORM_MEAN,  # pointer to the intermediate mean
    NORM_VAR,  # pointer to the intermediate variance
    N_COL,  # number of columns in X
    N_ROW,  # number of rows in X
    eps,  # a tiny number to prevent division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COL
    row_start = row_idx * N_COL
    X_row = X + row_start + col_offsets
    w_ptr = W + col_offsets
    b_ptr = B + col_offsets

    x = tl.load(X_row, mask=mask, other=0.0)
    weight = tl.load(w_ptr, mask=mask, other=1.0)
    bias = tl.load(b_ptr, mask=mask, other=0.0)

    # compute mean
    mean = tl.sum(x, axis=0) / N_COL
    # compute variance
    var = tl.sum((x - mean)**2, axis=0) / N_COL

    # layer norm
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_norm = x_norm * weight + bias

    # gelu
    output = 0.5 * x_norm * (1.0 + tl.tanh(0.7978845608028654 * x_norm * (1.0 + 0.044715 * x_norm * x_norm)))

    # write back the output
    Y_row = Y + row_start + col_offsets
    tl.store(Y_row, output, mask=mask)

    # store mean and var
    tl.store(NORM_MEAN + row_idx, mean)
    tl.store(NORM_VAR + row_idx, var)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROW, N_COL = x.shape
    output = torch.empty_like(x)
    norm_mean = torch.empty((N_ROW,), device=x.device, dtype=x.dtype)
    norm_var = torch.empty((N_ROW,), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(N_COL)
    grid = (N_ROW,)
    _layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        norm_mean,
        norm_var,
        N_COL,
        N_ROW,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output