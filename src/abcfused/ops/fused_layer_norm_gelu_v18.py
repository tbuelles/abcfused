# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    X, W, B, Y,
    N_ELEMENTS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N_ELEMENTS
    variance = tl.sum((x - mean) * (x - mean), axis=0) / N_ELEMENTS
    x_norm = (x - mean) / tl.sqrt(variance + eps)

    w = tl.load(W + offsets, mask=mask, other=1).to(tl.float32)
    b = tl.load(B + offsets, mask=mask, other=0).to(tl.float32)
    x_norm = x_norm * w + b

    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    tl.store(Y + offsets, gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    output = torch.empty_like(x)
    N_ELEMENTS = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    grid = (1, )
    _fused_layer_norm_gelu_kernel[grid](
        x, weight, bias, output,
        N_ELEMENTS=N_ELEMENTS,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return output