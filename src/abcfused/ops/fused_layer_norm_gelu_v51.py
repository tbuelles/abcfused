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
    eps,  # a small number to prevent division by zero
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    # Load data
    x = tl.load(X + row_idx * N_COLS + col_offsets, mask=mask, other=0.0)
    w = tl.load(W + col_offsets, mask=mask, other=0.0)
    b = tl.load(B + col_offsets, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS

    # Store mean and variance
    tl.store(Mean + row_idx, mean)
    tl.store(Var + row_idx, variance)

    # Normalize
    x_hat = (x - mean) / tl.sqrt(variance + eps)

    # Scale and bias
    x_hat = x_hat * w + b

    # Apply GELU activation
    output = x_hat * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))

    # Store the result
    tl.store(Y + row_idx * N_COLS + col_offsets, output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    mean = torch.empty(n_rows, device=x.device)
    variance = torch.empty(n_rows, device=x.device)

    grid = (n_rows,)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        variance,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=min(n_cols, 1024),
    )

    return output