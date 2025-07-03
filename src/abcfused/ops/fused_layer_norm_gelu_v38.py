# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B,
    Y,
    N_ELEMENTS: tl.constexpr,
    eps: tl.constexpr,
    ACTIVATION: tl.constexpr,
    VAR_TYPE: tl.constexpr,
    MEAN_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    x = tl.load(X + offsets, mask=mask)

    mean = tl.sum(x, axis=0) / N_ELEMENTS
    variance = tl.sum((x - mean) * (x - mean), axis=0) / N_ELEMENTS
    x_norm = (x - mean) / tl.sqrt(variance + eps)

    w = tl.load(W + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)

    x_norm = x_norm * w + b

    if ACTIVATION == 0:  # GELU
      output = x_norm * 0.5 * (1.0 + tl.erf(x_norm / tl.sqrt(2.0)))
    else:  # RELU
        output = tl.maximum(x_norm, 0)

    tl.store(Y + offsets, output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    output = torch.empty_like(x)
    N_ELEMENTS = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    grid = (1,)
    _layer_norm_gelu_kernel[grid](
        x, weight, bias,
        output,
        N_ELEMENTS=N_ELEMENTS,
        eps=eps,
        ACTIVATION=0, # GELU
        VAR_TYPE=x.dtype,
        MEAN_TYPE=x.dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output