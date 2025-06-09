# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    X, W, B,
    Y,
    Mean, Var,
    N, # Number of rows
    M, # Number of columns
    eps,
    **meta
):
    row = tl.program_id(0)
    cols = tl.arange(0, meta['BLOCK_SIZE'])
    mask = cols < M
    X_row = X + row * M
    x = tl.load(X_row + cols, mask=mask)
    mean = tl.sum(x, axis=0) / M
    var = tl.sum((x - mean)**2, axis=0) / M

    x_hat = (x - mean) / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    x_norm = x_hat * w + b

    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    tl.store(Y + row * M + cols, gelu, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Var + row, var)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    mean = torch.empty((N,), device=x.device, dtype=x.dtype)
    var = torch.empty((N,), device=x.device, dtype=x.dtype)
    output = torch.empty_like(x)

    grid = (N,)
    BLOCK_SIZE = min(triton.next_power_of_2(M), 2048)
    _fused_layer_norm_gelu_kernel[grid](
        x, weight, bias,
        output,
        mean, var,
        N, M, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output, mean, var