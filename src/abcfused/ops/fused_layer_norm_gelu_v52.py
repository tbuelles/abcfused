# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weight
    B,  # pointer to the bias
    MEAN,  # pointer to the mean
    VAR,  # pointer to the variance
    N_ELEMENTS,  # number of elements in X
    N_COLS,  # number of columns in X
    eps,  # a tiny number to avoid dividing by zero
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Map the block of rows that this instance of the kernel will compute to the
    # corresponding positions in the input X and output Y. We can do this in a
    # vectorized way, so we don't need to explicitly iterate through the rows.
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptr = X + row_idx * N_COLS + col_offsets
    # Load the data from x
    x = tl.load(input_ptr, mask=col_offsets < N_COLS, other=0.)
    # Compute mean
    mean = tl.sum(x, axis=0) / N_COLS
    # Compute variance
    var = tl.sum((x - mean)**2, axis=0) / N_COLS
    # Normalize
    x_hat = (x - mean) / tl.sqrt(var + eps)
    # Scale and shift
    w = tl.load(W + col_offsets, mask=col_offsets < N_COLS, other=1.)
    b = tl.load(B + col_offsets, mask=col_offsets < N_COLS, other=0.)
    x_hat = x_hat * w + b
    # Gelu
    output = x_hat * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))

    tl.store(Y + row_idx * N_COLS + col_offsets, output, mask=col_offsets < N_COLS)
    tl.store(MEAN + row_idx, mean)
    tl.store(VAR + row_idx, var)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    # Preallocate the output
    mean = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    var = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)
    N_ELEMENTS = x.numel()
    N_COLS = x.shape[-1]
    # Launch the kernel
    grid = (x.shape[0],)
    _fused_layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        var,
        N_ELEMENTS,
        N_COLS,
        eps,
        BLOCK_SIZE=1024,
    )
    return output