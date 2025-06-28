# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B, Y,
    N, M,
    eps,
    # layer norm
    mean_ptr,
    rstd_ptr,
    # fused relu
    gelu,
    # tensor strides
    X_stride_0, X_stride_1,
    W_stride_0,
    Y_stride_0, Y_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < M
    x_ptrs = X + row_idx * X_stride_0 + cols * X_stride_1
    x = tl.load(x_ptrs, mask=mask, other=0.).to(tl.float32)

    # --- layer norm ---
    mean = tl.sum(x, axis=0) / M
    var = tl.sum((x - mean) * (x - mean), axis=0) / M
    rstd = 1 / tl.sqrt(var + eps)
    x_norm = (x - mean) * rstd

    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)

    w_ptrs = W + cols * W_stride_0
    w = tl.load(w_ptrs, mask=mask, other=1.).to(tl.float32)
    b_ptrs = B + cols
    b = tl.load(b_ptrs, mask=mask, other=0.).to(tl.float32)
    x_norm = x_norm * w + b

    # --- gelu ---
    if gelu:
        x_norm = x_norm * 0.5 * (1.0 + tl.erf(x_norm / tl.sqrt(2.0)))

    y_ptrs = Y + row_idx * Y_stride_0 + cols * Y_stride_1
    tl.store(y_ptrs, x_norm.to(X.dtype), mask=mask)

def layer_norm_gelu(x, weight, bias, eps=1e-5, gelu=True):
    N, M = x.shape
    mean = torch.empty((N,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N,), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(M)
    grid = (N,)

    _layer_norm_gelu_kernel[grid](
        x, weight, bias, y,
        N, M,
        eps,
        mean,
        rstd,
        gelu,
        x.stride(0), x.stride(1),
        weight.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y