# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B, Output,
    Mean, Variance,
    N_COLS,
    eps,
    stride_xn, stride_wn, stride_bn, stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols_offsets = tl.arange(0, BLOCK_SIZE)
    mask = cols_offsets < N_COLS
    x = tl.load(X + row_idx * stride_xn + cols_offsets, mask=mask, other=0.0)

    # LayerNorm
    mean = tl.sum(x, axis=0) / N_COLS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    x_norm = (x - mean) / tl.sqrt(variance + eps)

    # GELU
    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Scale and shift
    w = tl.load(W + cols_offsets, mask=mask, other=1.0)
    b = tl.load(B + cols_offsets, mask=mask, other=0.0)
    output = x_norm * w + b + gelu

    tl.store(Output + row_idx * stride_on + cols_offsets, output, mask=mask)
    tl.store(Mean + row_idx, mean)
    tl.store(Variance + row_idx, variance)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    output = torch.empty_like(x)
    mean = torch.empty((N_ROWS,), dtype=torch.float32, device=x.device)
    variance = torch.empty((N_ROWS,), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N_COLS)
    if BLOCK_SIZE > 2048:
        BLOCK_SIZE = 2048
    _layer_norm_gelu_kernel[(N_ROWS,)](
        x, weight, bias, output,
        mean, variance,
        N_COLS,
        eps,
        x.stride(0), weight.stride(0), bias.stride(0), output.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output, mean, variance