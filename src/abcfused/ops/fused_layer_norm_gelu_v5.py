# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    Y,
    X,
    W,
    B,
    MEAN,
    RSTD,
    N_ELEMENTS,
    NORM_M,
    NORM_N,
    ACCUM_AXIS,
    BIAS,
    X_stride_0,
    X_stride_1,
    Y_stride_0,
    Y_stride_1,
    W_stride_0,
    B_stride_0,
    MEAN_stride_0,
    RSTD_stride_0,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    row_start = row_idx * BLOCK_SIZE
    col_start = col_idx * BLOCK_SIZE

    x_ptrs = X + row_start * X_stride_0 + col_start * X_stride_1
    w_ptr = W + col_start * W_stride_0
    b_ptr = B + col_start * B_stride_0
    mean_ptr = MEAN + row_idx * MEAN_stride_0
    rstd_ptr = RSTD + row_idx * RSTD_stride_0
    y_ptrs = Y + row_start * Y_stride_0 + col_start * Y_stride_1

    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_SIZE)[:, None] < NORM_M & tl.arange(0, BLOCK_SIZE)[None, :] < NORM_N, other=0.0)
    mean = tl.load(mean_ptr + tl.arange(0, 1), mask=tl.arange(0, 1) < 1, other=0.0)
    rstd = tl.load(rstd_ptr + tl.arange(0, 1), mask=tl.arange(0, 1) < 1, other=0.0)
    weight = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < NORM_N, other=0.0)
    bias = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < NORM_N, other=0.0)

    x_norm = (x - mean) * rstd
    x_scaled = x_norm * weight + bias
    gelu_out = x_scaled * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * x_scaled * (1.0 + 0.044715 * x_scaled * x_scaled)))
    tl.store(y_ptrs, gelu_out, mask=tl.arange(0, BLOCK_SIZE)[:, None] < NORM_M & tl.arange(0, BLOCK_SIZE)[None, :] < NORM_N)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    M, N = x.shape
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)

    norm_m = x.shape[0]
    norm_n = x.shape[1]

    n_elements = x.numel()
    accum_axis = 1 if x.is_contiguous() else 0
    bias_input = 1 if bias is not None else 0
    grid = (M, triton.cdiv(N, 16))

    _kernel[grid](
        y,
        x,
        weight,
        bias,
        mean,
        rstd,
        n_elements,
        norm_m,
        norm_n,
        accum_axis,
        bias_input,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        weight.stride(0),
        bias.stride(0),
        mean.stride(0),
        rstd.stride(0),
        BLOCK_SIZE=16,
        EPS=eps
    )

    return y