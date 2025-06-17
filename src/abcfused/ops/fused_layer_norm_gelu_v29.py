# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    X, W, B, Y,
    N_ELEMENTS: tl.constexpr,
    MEAN_ELEMENTS: tl.constexpr,
    VAR_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_idx = pid

    row_start = row_idx * N_ELEMENTS
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X + row_start + offsets, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / MEAN_ELEMENTS
    variance = tl.sum((x - mean)**2, axis=0) / VAR_ELEMENTS
    x_norm = (x - mean) / tl.sqrt(variance + eps)
    w = tl.load(W + offsets, mask=mask, other=1.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    x_norm = x_norm * w + b
    output = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
    tl.store(Y + row_start + offsets, output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    MEAN_ELEMENTS = N_ELEMENTS
    VAR_ELEMENTS = N_ELEMENTS
    if BLOCK_SIZE > 2048:
        BLOCK_SIZE = 2048
    output = torch.empty_like(x)
    grid = (x.shape[0],)
    _fused_layer_norm_gelu_kernel[grid](
        x, weight, bias, output,
        N_ELEMENTS=N_ELEMENTS,
        MEAN_ELEMENTS=MEAN_ELEMENTS,
        VAR_ELEMENTS=VAR_ELEMENTS,
        BLOCK_SIZE=BLOCK_SIZE,
        eps=eps,
    )
    return output