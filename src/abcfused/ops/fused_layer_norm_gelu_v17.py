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
    rstd,  # pointer to the 1/std
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load data, apply mask
    x = tl.load(X + row_idx * N + offsets, mask=mask, other=0.0)

    # Compute mean
    x_mean = tl.sum(x, axis=0) / N
    tl.store(mean + row_idx, x_mean)

    # Compute variance
    x_var = tl.sum((x - x_mean) * (x - x_mean), axis=0) / N

    # Compute standard deviation
    x_std = tl.sqrt(x_var + eps)
    rstd_value = 1.0 / x_std
    tl.store(rstd + row_idx, rstd_value)

    # Normalize
    x_norm = (x - x_mean) * rstd_value

    # Weight and bias
    w = tl.load(W + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    x_scaled = x_norm * w + b

    # GELU
    gelu_output = 0.5 * x_scaled * (1 + tl.tanh(0.7978845608028654 * x_scaled * (1 + 0.044715 * x_scaled * x_scaled)))

    # Store the result
    tl.store(Y + row_idx * N + offsets, gelu_output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N = x.shape[-1]
    assert x.is_contiguous()
    mean = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (x.shape[0],)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        rstd,
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output