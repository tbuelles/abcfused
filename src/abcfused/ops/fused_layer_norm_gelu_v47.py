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
    NORM_MEAN, # pointer to the mean
    NORM_VAR, # pointer to the variance
    N_ELEMENTS,  # number of elements in the vector
    # model arguments
    eps,  # epsilon to avoid division by zero
    N,  # matrix width
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X + offsets, mask=mask)

    # layer norm
    mean = tl.sum(x, axis=0) / N
    variance = tl.sum((x - mean)**2, axis=0) / N

    x_norm = (x - mean) / tl.sqrt(variance + eps)
    w = tl.load(W + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    x_norm = x_norm * w + b
    
    # gelu
    gelu_val = 0.5 * x_norm * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.pi) * (x_norm + 0.044715 * x_norm**3)))

    tl.store(Y + offsets, gelu_val, mask=mask)
    tl.store(NORM_MEAN + row_idx, mean)
    tl.store(NORM_VAR + row_idx, variance)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.shape[-1]
    N = N_ELEMENTS
    output = torch.empty_like(x)
    norm_mean = torch.empty(1, dtype=torch.float32, device=x.device) # Modified to float32
    norm_var = torch.empty(1, dtype=torch.float32, device=x.device) # Modified to float32
    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    grid = (1, )

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        norm_mean,
        norm_var,
        N_ELEMENTS,
        eps,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output