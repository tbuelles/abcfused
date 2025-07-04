# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B, R,
    Mean, Variance,
    N_ROWS, N_COLS,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    row = X + row_idx * N_COLS
    x = tl.load(row + col_offsets, mask=mask, other=0.)

    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) * (x - mean), axis=0) / N_COLS

    x_norm = (x - mean) / tl.sqrt(variance + eps)

    w = tl.load(W + col_offsets, mask=mask, other=1.)
    b = tl.load(B + col_offsets, mask=mask, other=0.)

    x_norm = x_norm * w + b

    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    r = R + row_idx * N_COLS
    tl.store(r + col_offsets, gelu, mask=mask)

    mean_ptr = Mean + row_idx
    var_ptr = Variance + row_idx
    tl.store(mean_ptr, mean)
    tl.store(var_ptr, variance)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    R = torch.empty_like(x)
    Mean = torch.empty((N_ROWS,), device=x.device, dtype=x.dtype)
    Variance = torch.empty((N_ROWS,), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(N_COLS)
    grid = (N_ROWS,)

    _layer_norm_gelu_kernel[grid](
        x, weight, bias, R,
        Mean, Variance,
        N_ROWS, N_COLS,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return R