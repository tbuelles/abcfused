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
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= N_ROWS:
        return

    cols_range = tl.arange(0, BLOCK_SIZE)
    X_ptr = X + row_idx * N_COLS + cols_range
    W_ptr = W + cols_range
    B_ptr = B + cols_range

    x = tl.load(X_ptr, mask=cols_range < N_COLS, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N_COLS

    # Compute variance
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS

    # Store mean and variance for potential backward pass
    tl.store(Mean + row_idx, mean)
    tl.store(Var + row_idx, variance)

    # Normalize
    x_norm = (x - mean) / tl.sqrt(variance + eps)

    # Apply weights and biases
    w = tl.load(W_ptr, mask=cols_range < N_COLS, other=1.0)
    b = tl.load(B_ptr, mask=cols_range < N_COLS, other=0.0)
    x_norm = x_norm * w + b

    # Apply GELU activation
    gelu_out = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Store the result
    Y_ptr = Y + row_idx * N_COLS + cols_range
    tl.store(Y_ptr, gelu_out, mask=cols_range < N_COLS)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    mean = torch.empty((N_ROWS,), device=x.device, dtype=torch.float32)
    var = torch.empty((N_ROWS,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N_COLS)
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
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output