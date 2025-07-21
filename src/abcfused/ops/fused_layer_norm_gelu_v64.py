# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    X, W, B, Mean, Variance, Output,
    N_ROWS, N_COLS,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS
    row = X + row_idx * N_COLS
    x = tl.load(row + col_offsets, mask=mask, other=0.).to(tl.float32)

    # LayerNorm
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) * (x - mean), axis=0) / N_COLS
    x_norm = (x - mean) / tl.sqrt(variance + eps)

    # GELU
    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Scale and shift
    w = tl.load(W + col_offsets, mask=mask, other=1.).to(tl.float32)
    b = tl.load(B + col_offsets, mask=mask, other=0.).to(tl.float32)
    output = gelu * w + b

    tl.store(Mean + row_idx, mean)
    tl.store(Variance + row_idx, variance)
    tl.store(Output + row_idx * N_COLS + col_offsets, output, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    output = torch.empty_like(x)
    mean = torch.empty((N_ROWS,), dtype=torch.float32, device=x.device)
    variance = torch.empty((N_ROWS,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N_COLS)
    if BLOCK_SIZE > 2048:
        BLOCK_SIZE = 2048
    _fused_layer_norm_gelu_kernel[(N_ROWS,)](
        x, weight, bias, mean, variance, output,
        N_ROWS, N_COLS,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output