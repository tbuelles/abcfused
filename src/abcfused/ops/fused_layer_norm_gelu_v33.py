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
    rstd,  # pointer to the rstd
    n_cols,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    N,  # batch dimension
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    X_row_ptr = X + row_idx * n_cols + col_offsets
    x = tl.load(X_row_ptr, mask=mask, other=0.0)

    # compute mean
    sum_x = tl.sum(x, axis=0)
    mean_output = sum_x / n_cols
    mean_ptr = mean + row_idx
    tl.store(mean_ptr, mean_output)

    # compute variance
    var = tl.sum(tl.square(x - mean_output), axis=0) / n_cols
    rstd_output = 1 / tl.sqrt(var + eps)
    rstd_ptr = rstd + row_idx
    tl.store(rstd_ptr, rstd_output)

    # layer norm
    x_norm = (x - mean_output) * rstd_output
    W_ptr = W + col_offsets
    B_ptr = B + col_offsets
    w = tl.load(W_ptr, mask=mask, other=1.0)
    b = tl.load(B_ptr, mask=mask, other=0.0)
    x_norm = x_norm * w + b

    # gelu
    output = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * x_norm * (1.0 + 0.044715 * x_norm * x_norm)))

    Y_row_ptr = Y + row_idx * n_cols + col_offsets
    tl.store(Y_row_ptr, output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, n_cols = x.shape
    mean = torch.empty((N,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (N,)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        rstd,
        n_cols,
        eps,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output