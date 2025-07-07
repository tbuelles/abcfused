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
    N_ELEMENTS,  # number of elements in the vector
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_indices = row_start + tl.arange(0, BLOCK_SIZE)
    mask = row_indices < N_ELEMENTS

    x = tl.load(X + row_indices, mask=mask, other=0.0)
    weight = tl.load(W + row_indices, mask=mask, other=1.0)
    bias = tl.load(B + row_indices, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N_ELEMENTS
    variance = tl.sum((x - mean)**2, axis=0) / N_ELEMENTS

    x_norm = (x - mean) / tl.sqrt(variance + eps)
    output = x_norm * weight + bias

    gelu_output = 0.5 * output * (1.0 + tl.tanh(0.7978845608028654 * (output + 0.044715 * output * output * output)))

    tl.store(Y + row_indices, gelu_output, mask=mask)
    tl.store(MEAN + pid, mean)
    tl.store(VAR + pid, variance)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.shape[-1]
    output = torch.empty_like(x)
    mean = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    var = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    grid = (x.shape[0],)
    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        var,
        N_ELEMENTS,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output