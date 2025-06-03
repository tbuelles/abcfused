# type: ignore

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd, # pointer to the variance
    N_ELEMENTS: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, GROUP_SIZE)
    mask = cols < N_ELEMENTS

    # Load data
    x = tl.load(X + row_idx * N_ELEMENTS + cols, mask=mask, other=0.0)

    # Calculate mean and variance
    mean = tl.sum(x, axis=0) / N_ELEMENTS
    variance = tl.sum((x - mean)**2, axis=0) / N_ELEMENTS
    rstd = 1 / tl.sqrt(variance + eps)

    # Store mean and variance
    tl.store(Mean + row_idx, mean)
    tl.store(Rstd + row_idx, rstd)

    # Normalize
    x_norm = (x - mean) * rstd

    # Load weights and biases
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # Scale and shift
    x_scaled = x_norm * w + b

    # GELU activation
    output = x_scaled * 0.5 * (1.0 + tl.tanh(0.7978845608 * x_scaled * (1.0 + 0.044715 * x_scaled * x_scaled)))

    # Store output
    tl.store(Y + row_idx * N_ELEMENTS + cols, output, mask=mask)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    output = torch.empty_like(x)
    mean = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    rstd = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    N_ELEMENTS = x.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    grid = (x.shape[0],)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        rstd,
        N_ELEMENTS=N_ELEMENTS,
        eps=eps,
        GROUP_SIZE=BLOCK_SIZE,
    )

    return output