# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_fwd_kernel(
    X,  # data ptr
    Y,  # output ptr
    W,  # weight ptr
    B,  # bias ptr
    mean,  # mean ptr
    rstd,  # rstd ptr
    N,  # number of rows
    M,  # number of columns
    eps,  # layer norm epsilon
    **meta
):
    row = tl.program_id(0)
    cols = tl.arange(0, meta['BLOCK_SIZE'])
    mask = cols < M

    # Compute mean and variance
    x = tl.load(X + row * M + cols, mask=mask, other=0.0)
    sum_x = tl.sum(x, axis=0)
    tl.store(mean + row, sum_x / M)

    x_mean = tl.load(mean + row)
    x_centered = x - x_mean
    sum_x2 = tl.sum(x_centered * x_centered, axis=0)
    var = sum_x2 / M
    tl.store(rstd + row, 1 / tl.sqrt(var + eps))

    # Normalize, apply weight and bias, then GELU
    x_norm = x_centered * tl.load(rstd + row)
    weight = tl.load(W + cols, mask=mask, other=1.0)
    bias = tl.load(B + cols, mask=mask, other=0.0)
    x_scaled = x_norm * weight + bias

    gelu = x_scaled * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * x_scaled * (1.0 + 0.044715 * x_scaled * x_scaled)))
    tl.store(Y + row * M + cols, gelu, mask=mask)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    mean = torch.empty((N,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    grid = (N,)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        rstd,
        N,
        M,
        eps,
        BLOCK_SIZE=1024,
    )
    return output