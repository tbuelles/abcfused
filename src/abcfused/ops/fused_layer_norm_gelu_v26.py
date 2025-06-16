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
    N_ELEMENTS: tl.constexpr,  # number of elements in the input
    MEAN,  # pointer to intermediate mean
    RVAR,  # pointer to intermediate rvar
    eps: tl.constexpr,  # epsilon to avoid division by zero
    VAR_SCALING_FACTOR: tl.constexpr, # scaling factor for variance (e.g. (N-1)/N)
    BLOCK_SIZE: tl.constexpr,  # size of the block
):
    row_idx = tl.program_id(0)
    offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load data, compute mean and variance
    x = tl.load(X + offsets, mask=mask)
    mean = tl.sum(x, axis=0) / N_ELEMENTS
    var = tl.sum((x - mean) * (x - mean), axis=0) / N_ELEMENTS * VAR_SCALING_FACTOR

    # Store mean and variance
    tl.store(MEAN + row_idx, mean)
    tl.store(RVAR + row_idx, var)

    # Normalize and apply linear transformation
    x_hat = (x - mean) / tl.sqrt(var + eps)
    w = tl.load(W + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    y = x_hat * w + b

    # Apply GELU activation
    gelu_val = 0.5 * y * (1.0 + tl.tanh(0.7978845608028654 * (y + 0.044715 * y * y * y)))

    # Store the result
    tl.store(Y + offsets, gelu_val, mask=mask)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.shape[-1]
    output = torch.empty_like(x)
    mean = torch.empty(x.shape[0], device=x.device)
    rvar = torch.empty(x.shape[0], device=x.device)
    BLOCK_SIZE = N_ELEMENTS
    grid = (x.shape[0],)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        N_ELEMENTS,
        mean,
        rvar,
        eps,
        (N_ELEMENTS-1)/N_ELEMENTS,
        BLOCK_SIZE,
        num_warps=4,
    )
    return output