# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_fwd_kernel(
    X,  # data ptr
    Y,  # output ptr
    W,  # weight ptr
    B,  # bias ptr
    Mean,  # mean ptr
    Var,  # variance ptr
    N_ROWS,  # number of rows
    N_COLS,  # number of cols
    eps,  # layer norm epsilon
    # BLOCK_SIZE: tl.constexpr,
    **meta
):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']
    # Map the row index to the elements we want to compute
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS

    # Load data, weight, and bias
    x = tl.load(X + row_idx * N_COLS + offsets, mask=mask, other=0).to(tl.float32)
    weight = tl.load(W + offsets, mask=mask, other=1).to(tl.float32)
    bias = tl.load(B + offsets, mask=mask, other=0).to(tl.float32)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS

    tl.store(Mean + row_idx, mean)
    tl.store(Var + row_idx, variance)

    # Normalize
    x_hat = (x - mean) / tl.sqrt(variance + eps)

    # Scale and shift
    x_hat = x_hat * weight + bias

    # Apply GELU activation function
    output = x_hat * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))

    # Store the result
    tl.store(Y + row_idx * N_COLS + offsets, output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    BLOCK_SIZE = N_COLS
    output = torch.empty_like(x)
    mean = torch.empty((N_ROWS,), dtype=torch.float32, device="cuda")
    var = torch.empty((N_ROWS,), dtype=torch.float32, device="cuda")

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

    return output, mean, var