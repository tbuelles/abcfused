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
    Mean,  # pointer to the mean
    Var,  # pointer to the variance
    N_ROWS,  # number of rows in X
    N_COLS,  # number of columns in X
    eps,  # epsilon to avoid dividing by zero
    **meta
):
    row_idx = tl.program_id(0)
    if row_idx >= N_ROWS:
        return

    cols_range = tl.arange(0, N_COLS)
    x = tl.load(X + row_idx * N_COLS + cols_range)

    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS

    x_norm = (x - mean) / tl.sqrt(variance + eps)
    w = tl.load(W + cols_range)
    b = tl.load(B + cols_range)
    x_norm = x_norm * w + b

    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    tl.store(Y + row_idx * N_COLS + cols_range, gelu)
    tl.store(Mean + row_idx, mean)
    tl.store(Var + row_idx, variance)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    mean = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
    var = torch.empty(N_ROWS, device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    grid = (N_ROWS,)
    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        var,
        N_ROWS,
        N_COLS,
        eps,
        num_warps=4,
        num_stages=1,
    )

    return output