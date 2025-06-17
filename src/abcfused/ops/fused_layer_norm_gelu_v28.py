# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B,
    Y,
    N_ELEMENTS: tl.constexpr,
    eps: tl.constexpr,
    MEAN,
    RSTD,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, N_ELEMENTS)

    X_row = X + row_idx * N_ELEMENTS
    W_row = W
    B_row = B
    Y_row = Y + row_idx * N_ELEMENTS

    x = tl.load(X_row + cols, mask=cols < N_ELEMENTS, other=0.0)

    mean = tl.load(MEAN + row_idx)
    rstd = tl.load(RSTD + row_idx)

    x_norm = (x - mean) * rstd

    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    w = tl.load(W_row + cols, mask=cols < N_ELEMENTS, other=1.0)
    b = tl.load(B_row + cols, mask=cols < N_ELEMENTS, other=0.0)

    output = gelu * w + b
    tl.store(Y_row + cols, output, mask=cols < N_ELEMENTS)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.shape[-1]
    rows = x.shape[0]
    output = torch.empty_like(x)

    mean = torch.empty((rows,), device=x.device)
    rstd = torch.empty((rows,), device=x.device)

    x_f = x.float()
    mean[:] = torch.mean(x_f, dim=1)
    rstd[:] = torch.sqrt(torch.var(x_f, dim=1) + eps).pow(-1)

    grid = (rows,)
    _layer_norm_gelu_kernel[grid](
        x, weight, bias,
        output,
        N_ELEMENTS=N_ELEMENTS,
        eps=eps,
        MEAN=mean,
        RSTD=rstd
    )

    return output