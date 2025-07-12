# type: ignore

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_layer_norm_gelu_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    NORM_MEAN,  # pointer to the mean
    NORM_VAR,  # pointer to the variance
    N_ELEMENTS,  # number of elements in the vector
    EPS: tl.float32,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.int32,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X + offsets, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N_ELEMENTS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_ELEMENTS

    x_norm = (x - mean) / tl.sqrt(variance + EPS)

    w = tl.load(W + offsets, mask=mask, other=1.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    x_norm = x_norm * w + b

    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    tl.store(Y + offsets, gelu, mask=mask)
    tl.store(NORM_MEAN + pid, mean)
    tl.store(NORM_VAR + pid, variance)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    num_rows = x.shape[0]

    output = torch.empty_like(x)
    norm_mean = torch.empty((num_rows,), device=x.device, dtype=torch.float32)
    norm_var = torch.empty((num_rows,), device=x.device, dtype=torch.float32)

    grid = (num_rows,)

    _fused_layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        norm_mean,
        norm_var,
        N_ELEMENTS,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output