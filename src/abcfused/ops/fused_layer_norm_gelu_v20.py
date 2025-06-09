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
    Variance,  # pointer to the variance
    N_ROWS,  # number of rows in X
    N_COLS,  # number of columns in X
    eps,  # a tiny number to prevent division by zero
    # scale,
    # pointer to output,
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program ID to the row of X and Y that it will compute.
    row_idx = tl.program_id(0)
    # The stride represents how far ahead each pointer is when you move it by 1 position.
    row_start_ptr = X + row_idx * N_COLS
    # Initialize pointers to the row of X and Y that the program will compute.
    row_ptrs = row_start_ptr + tl.arange(0, BLOCK_SIZE)
    # Load the row of X into registers.
    x = tl.load(row_ptrs, mask=row_ptrs < row_start_ptr + N_COLS).to(tl.float32)
    # Compute mean and variance.
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    # Normalize.
    x_hat = (x - mean) / tl.sqrt(variance + eps)
    # Scale and shift.
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N_COLS).to(tl.float32)
    b = tl.load(B + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N_COLS).to(tl.float32)
    x_hat = x_hat * w + b

    # Apply GELU activation
    output = x_hat * 0.5 * (1.0 + tl.erf(x_hat / 1.41421356237))

    # Write the output to Y.
    output_ptrs = Y + row_idx * N_COLS + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptrs, output, mask=output_ptrs < Y + (row_idx + 1) * N_COLS)
    # Write the mean and variance to Mean and Variance.
    tl.store(Mean + row_idx, mean)
    tl.store(Variance + row_idx, variance)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    mean = torch.empty((N_ROWS,), device=x.device, dtype=torch.float32)
    variance = torch.empty((N_ROWS,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    grid = (N_ROWS,)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        variance,
        N_ROWS,
        N_COLS,
        eps,
        BLOCK_SIZE=N_COLS,
    )
    return output