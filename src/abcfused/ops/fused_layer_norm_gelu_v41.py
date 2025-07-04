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
    MEAN,  # pointer to the mean
    VAR,  # pointer to the variance
    N_ELEMENTS: tl.constexpr,  # number of elements in the input
    N_COLS: tl.constexpr,  # number of columns in the input
    eps: tl.constexpr,  # epsilon to avoid division by zero
):
    row_idx = tl.program_id(0)
    cols_range = tl.arange(0, N_COLS)

    # Load data
    x = tl.load(X + row_idx * N_COLS + cols_range, mask=cols_range < N_COLS, other=0.0)

    # Calculate mean
    mean = tl.sum(x, axis=0) / N_COLS
    tl.store(MEAN + row_idx, mean)

    # Calculate variance
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    tl.store(VAR + row_idx, variance)

    # Normalize
    x_hat = (x - mean) / tl.sqrt(variance + eps)

    # Apply weights and biases
    w = tl.load(W + cols_range, mask=cols_range < N_COLS, other=1.0)
    b = tl.load(B + cols_range, mask=cols_range < N_COLS, other=0.0)
    x_normed = x_hat * w + b

    # Apply GELU
    gelu_out = 0.5 * x_normed * (1.0 + tl.tanh(0.7978845608028654 * (x_normed + 0.044715 * x_normed * x_normed * x_normed)))

    # Store the result
    tl.store(Y + row_idx * N_COLS + cols_range, gelu_out, mask=cols_range < N_COLS)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS, N_COLS = x.shape
    output = torch.empty_like(x)
    mean = torch.empty((N_ELEMENTS,), dtype=torch.float32, device=x.device)
    variance = torch.empty((N_ELEMENTS,), dtype=torch.float32, device=x.device)

    grid = (N_ELEMENTS,)
    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        variance,
        N_ELEMENTS=N_ELEMENTS,
        N_COLS=N_COLS,
        eps=eps,
    )

    return output