# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    N_COLS,  # number of columns in X
    eps,  # a tiny number to prevent division by zero
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Map the thread program ID to the elements of X and Y it should compute.
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = X + row_idx * N_COLS + col_offsets
    # Compute mean
    x = tl.load(x_ptrs, mask=col_offsets < N_COLS, other=0.)
    mean = tl.sum(x, axis=0) / N_COLS
    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    # Normalize
    norm = (x - mean) / tl.sqrt(var + eps)
    # Multiply by weight and add bias
    w = tl.load(W + col_offsets, mask=col_offsets < N_COLS, other=1.)
    b = tl.load(B + col_offsets, mask=col_offsets < N_COLS, other=0.)
    norm = norm * w + b
    # Apply GELU
    gelu = norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (norm + 0.044715 * norm * norm * norm)))

    # Write back the block of the output tensor
    y_ptrs = Y + row_idx * N_COLS + col_offsets
    tl.store(y_ptrs, gelu, mask=col_offsets < N_COLS)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    output = torch.empty_like(x)
    N_ROWS, N_COLS = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N_COLS)
    grid = (N_ROWS,)
    _layer_norm_gelu_kernel[grid](
        output,
        x,
        weight,
        bias,
        N_COLS,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output