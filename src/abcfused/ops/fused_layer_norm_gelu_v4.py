# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    Y,  # output pointer
    X,  # input pointer
    W,  # weight pointer
    B,  # bias pointer
    N_ELEMENTS: tl.constexpr,
    MEAN: tl.constexpr,
    RSTD: tl.constexpr,
    NORM_M: tl.constexpr,
    NORM_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offset = row_idx * NORM_N
    cols = tl.arange(0, NORM_N)
    mask = cols < NORM_N

    x = tl.load(X + offset + cols, mask=mask, other=0.0)
    mean = tl.load(MEAN + row_idx, mask=mask, other=0.0)
    rstd = tl.load(RSTD + row_idx, mask=mask, other=0.0)
    weight = tl.load(W + cols, mask=mask, other=0.0)
    bias = tl.load(B + cols, mask=mask, other=0.0)

    # Layer Norm
    x_norm = (x - mean) * rstd
    x_norm = x_norm * weight + bias

    # GELU
    gelu = 0.5 * x_norm * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    tl.store(Y + offset + cols, gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    M, N = x.shape
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
    torch.layer_norm(x, normalized_shape=[N], weight=weight, bias=bias, eps=eps, out=(mean, rstd))
    rstd = 1 / torch.sqrt(rstd)

    output = torch.empty_like(x)
    grid = (x.shape[0],)
    _kernel[grid](
        output,
        x,
        weight,
        bias,
        x.numel(),
        mean,
        rstd,
        x.shape[0],
        x.shape[1],
        num_warps=4,
    )
    return output