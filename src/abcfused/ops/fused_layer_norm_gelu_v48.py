# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B, Y,
    N, M,
    eps,
    mean, rstd,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < M
    X_row = X + row_idx * M
    w_ptr = W
    b_ptr = B

    x = tl.load(X_row + cols, mask=mask, other=0.0)

    # LayerNorm
    x_mean = tl.load(mean + row_idx, mask=True, other=0.0)
    x_rstd = tl.load(rstd + row_idx, mask=True, other=0.0)
    x_norm = (x - x_mean) * x_rstd

    # GELU
    gelu = 0.5 * x_norm * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Scale and shift
    weight = tl.load(w_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(b_ptr + cols, mask=mask, other=0.0)
    output = gelu * weight + bias

    Y_row = Y + row_idx * M
    tl.store(Y_row + cols, output, mask=mask)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    mean = torch.empty((N,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N,), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)

    x_mean = torch.mean(x, dim=1)
    x_var = torch.var(x, dim=1)
    x_rstd = 1 / torch.sqrt(x_var + eps)

    mean[:] = x_mean
    rstd[:] = x_rstd

    BLOCK_SIZE = triton.next_power_of_2(M)
    _layer_norm_gelu_kernel[(N,)](
        x, weight, bias, y,
        N, M,
        eps,
        mean, rstd,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y