# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    X, W, B, Y,
    N, M,
    eps,
    **meta
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, meta['BLOCK_SIZE'])
    mask = cols < M

    # X + LayerNorm
    x = tl.load(X + row_idx * M + cols, mask=mask)
    mean = tl.sum(x, axis=0) / M
    var = tl.sum((x - mean) * (x - mean), axis=0) / M
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std

    # LayerNorm + W*X + B
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    x = x_norm * w + b

    # Gelu
    cst = 0.7978845608028654
    gelu = 0.5 * x * (1.0 + tl.tanh(cst * x * (1.0 + 0.044715 * x * x)))

    tl.store(Y + row_idx * M + cols, gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    output = torch.empty_like(x)
    grid = (N,)
    BLOCK_SIZE = min(triton.next_power_of_2(M), 2048)
    _fused_layer_norm_gelu_kernel[grid](
        x, weight, bias, output,
        N, M,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output