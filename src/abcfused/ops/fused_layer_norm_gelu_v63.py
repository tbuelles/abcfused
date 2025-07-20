# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_layer_norm_gelu(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    mean,  # pointer to the mean
    var,  # pointer to the variance
    N,  # number of rows in X
    M,  # number of columns in X
    eps,  # a scalar value used for numerical stability
    gelu_approx, # Gelu approximation
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < M

    # compute mean
    row = tl.load(X + row_idx * M + cols, mask=mask, other=0.0)
    row_mean = tl.sum(row, axis=0) / M
    tl.store(mean + row_idx, row_mean)

    # compute variance
    row_var = tl.sum((row - row_mean) ** 2, axis=0) / M
    tl.store(var + row_idx, row_var)

    # layer norm
    inv_std = 1.0 / tl.sqrt(row_var + eps)
    normed = (row - row_mean) * inv_std
    weight = tl.load(W + cols, mask=mask, other=1.0)
    bias = tl.load(B + cols, mask=mask, other=0.0)
    output = normed * weight + bias

    # gelu
    if gelu_approx:
        output = 0.5 * output * (1.0 + tl.tanh(0.7978845608 * output * (1.0 + 0.044715 * output * output)))
    else:
        output = output * 0.5 * (1.0 + tl.erf(output / 1.41421356237))

    tl.store(Y + row_idx * M + cols, output, mask=mask)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5, gelu_approx=True):
    N, M = x.shape
    mean = torch.empty((N,), device=x.device, dtype=torch.float32)
    var = torch.empty((N,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    BLOCK_SIZE = min(triton.next_power_of_2(M), 2048)
    grid = (N,)

    _kernel_layer_norm_gelu[grid](
        x,
        output,
        weight,
        bias,
        mean,
        var,
        N,
        M,
        eps,
        gelu_approx,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output