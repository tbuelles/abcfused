# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    X, W, B, Y,
    N, M,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < M

    x = tl.load(X + row_idx * M + cols, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / M
    var = tl.sum((x - mean) * (x - mean), axis=0) / M

    x_norm = (x - mean) / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    x_norm = x_norm * w + b
    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    tl.store(Y + row_idx * M + cols, gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    output = torch.empty_like(x)

    grid = (N,)
    _fused_layer_norm_gelu_kernel[grid](
        x, weight, bias, output,
        N, M,
        eps,
        BLOCK_SIZE=M,
    )

    return output