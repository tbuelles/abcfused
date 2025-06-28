# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    MEAN, # pointer to the mean
    RSTD, # pointer to the rstd
    N_ELEMENTS: tl.constexpr,  # number of elements in the input
    ROW_SIZE: tl.constexpr,      # number of elements in a row
    EPS: tl.constexpr,          # epsilon to avoid division by zero
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, ROW_SIZE)
    mask = cols < ROW_SIZE

    # load row from X
    x = tl.load(X + row_idx * ROW_SIZE + cols, mask=mask, other=0.0)

    # compute mean and variance
    mean = tl.sum(x, axis=0) / ROW_SIZE
    variance = tl.sum((x - mean) ** 2, axis=0) / ROW_SIZE

    # compute rstd
    rstd = 1.0 / tl.sqrt(variance + EPS)

    # store mean and rstd
    tl.store(MEAN + row_idx, mean)
    tl.store(RSTD + row_idx, rstd)

    # normalize
    x_hat = (x - mean) * rstd

    # apply weights and biases
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    x_hat = x_hat * w + b

    # apply gelu
    gelu = x_hat * 0.5 * (1.0 + tl.erf(x_hat / tl.sqrt(2.0)))

    # store the result
    tl.store(Y + row_idx * ROW_SIZE + cols, gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.shape[0] * x.shape[1]
    ROW_SIZE = x.shape[1]
    output = torch.empty_like(x)
    mean = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)

    grid = (x.shape[0],)
    _layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        rstd,
        N_ELEMENTS=N_ELEMENTS,
        ROW_SIZE=ROW_SIZE,
        EPS=eps,
    )

    return output