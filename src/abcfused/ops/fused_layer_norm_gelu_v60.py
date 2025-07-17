# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_layer_norm_gelu_kernel(
    Y,  # pointer to output
    X,  # pointer to input
    W,  # pointer to weight
    B,  # pointer to bias
    mean,  # pointer to mean of input
    rstd,  # pointer to 1/std of input
    N_ELEMENTS: tl.constexpr,
    N_FEATURES: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_idx = pid

    row_start = row_idx * N_FEATURES
    offsets = tl.arange(0, N_FEATURES)
    mask = offsets < N_FEATURES

    x = tl.load(X + row_start + offsets, mask=mask, other=0.).to(tl.float32)

    # LayerNorm
    x_mean = tl.sum(x, axis=0) / N_FEATURES
    x_var = tl.sum((x - x_mean) * (x - x_mean), axis=0) / N_FEATURES
    x_std = tl.sqrt(x_var + eps)
    x_rstd = 1. / x_std

    x_normalized = (x - x_mean) * x_rstd
    w = tl.load(W + offsets, mask=mask, other=1.).to(tl.float32)
    b = tl.load(B + offsets, mask=mask, other=0.).to(tl.float32)
    x_scaled = x_normalized * w + b

    # GELU
    gelu_val = 0.5 * x_scaled * (1.0 + tl.tanh(0.7978845608028654 * (x_scaled + 0.044715 * x_scaled * x_scaled * x_scaled)))

    tl.store(Y + row_start + offsets, gelu_val, mask=mask)
    tl.store(mean + row_idx, x_mean)
    tl.store(rstd + row_idx, x_rstd)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS, N_FEATURES = x.shape
    mean = torch.empty((N_ELEMENTS,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N_ELEMENTS,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    grid = (N_ELEMENTS,)
    _fused_layer_norm_gelu_kernel[grid](
        output,
        x,
        weight,
        bias,
        mean,
        rstd,
        N_ELEMENTS=N_ELEMENTS,
        N_FEATURES=N_FEATURES,
        eps=eps,
    )
    return output