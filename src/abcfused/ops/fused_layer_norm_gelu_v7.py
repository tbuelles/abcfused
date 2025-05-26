# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    Y,
    X,
    W,
    B,
    MEAN,
    RVAR,
    N_ELEMENTS,
    eps,
    stride_y,
    stride_x,
    stride_w,
    stride_b,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X + offsets, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / N_ELEMENTS
    variance = tl.sum((x - mean) * (x - mean), axis=0) / N_ELEMENTS

    rvar = 1 / tl.sqrt(variance + eps)

    w = tl.load(W + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    x_norm = (x - mean) * rvar
    y = x_norm * w + b
    gelu = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(Y + offsets, gelu, mask=mask)
    tl.store(MEAN + row_idx, mean)
    tl.store(RVAR + row_idx, rvar)

def fused_layer_norm_gelu(y, x, w, b, eps=1e-5):
    n_elements = x.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    num_rows = x.shape[0]

    mean = torch.empty((num_rows,), device=x.device, dtype=torch.float32)
    rvar = torch.empty((num_rows,), device=x.device, dtype=torch.float32)

    grid = (num_rows,)

    _layer_norm_gelu_kernel[grid](
        y,
        x,
        w,
        b,
        mean,
        rvar,
        n_elements,
        eps,
        y.stride(0),
        x.stride(0),
        w.stride(0),
        b.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y