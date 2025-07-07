# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B,
    Out,
    Mean, Var,
    N_FEATURES,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_FEATURES
    x = tl.load(X + row_idx * N_FEATURES + col_offsets, mask=mask, other=0.)

    # Calculate mean and variance
    mean = tl.sum(x, axis=0) / N_FEATURES
    variance = tl.sum((x - mean) * (x - mean), axis=0) / N_FEATURES
    tl.store(Mean + row_idx, mean)
    tl.store(Var + row_idx, variance)

    # Layer norm
    x_hat = (x - mean) / tl.sqrt(variance + eps)
    w = tl.load(W + col_offsets, mask=mask, other=1.)
    b = tl.load(B + col_offsets, mask=mask, other=0.)
    x_normed = x_hat * w + b

    # GELU
    gelu_val = 0.5 * x_normed * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (x_normed + 0.044715 * x_normed * x_normed * x_normed)))

    tl.store(Out + row_idx * N_FEATURES + col_offsets, gelu_val, mask=mask)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_FEATURES = x.shape
    Out = torch.empty_like(x)
    Mean = torch.empty((N_ROWS,), device=x.device, dtype=torch.float32)
    Var = torch.empty((N_ROWS,), device=x.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(N_FEATURES)
    grid = (N_ROWS,)

    _layer_norm_gelu_kernel[grid](
        x, weight, bias,
        Out,
        Mean, Var,
        N_FEATURES,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return Out