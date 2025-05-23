# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    N,  # number of rows in X
    D,  # number of columns in X
    eps,  # epsilon to avoid dividing by zero
    # barrier synchronization object
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE_D)
    mask = col_offsets < D

    # compute mean
    x_ptrs = X + row_idx * D + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.)
    mean = tl.sum(x, axis=0) / D

    # compute variance
    x_minus_mean = x - mean
    variance = tl.sum(x_minus_mean * x_minus_mean, axis=0) / D

    # compute normalized input
    x_norm = (x - mean) / tl.sqrt(variance + eps)

    # weight and bias
    w_ptrs = W + col_offsets
    b_ptrs = B + col_offsets
    weight = tl.load(w_ptrs, mask=mask, other=1.)
    bias = tl.load(b_ptrs, mask=mask, other=0.)

    # layer norm output
    x_norm = x_norm * weight + bias

    # gelu activation
    output = 0.5 * x_norm * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # write back the block to the device
    y_ptrs = Y + row_idx * D + col_offsets
    tl.store(y_ptrs, output, mask=mask)

def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, D = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_D = 128
    grid = (N,)
    _layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        N,
        D,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return output