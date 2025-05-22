# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the bias
    MEAN,  # pointer to the mean
    VAR,  # pointer to the variance
    N_ELEMENTS: tl.constexpr,  # number of elements in the input
    ROW_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, ROW_SIZE)

    x = tl.load(X + row_idx * ROW_SIZE + cols, mask=cols < ROW_SIZE, other=0.0)

    # Layer Norm
    mean = tl.sum(x, axis=0) / ROW_SIZE
    variance = tl.sum((x - mean) * (x - mean), axis=0) / ROW_SIZE

    x_norm = (x - mean) / tl.sqrt(variance + eps)
    w = tl.load(W + cols, mask=cols < ROW_SIZE, other=1.0)
    b = tl.load(B + cols, mask=cols < ROW_SIZE, other=0.0)
    x_norm = x_norm * w + b

    # GELU
    gelu_val = 0.5 * x_norm * (1 + tl.tanh(tl.sqrt(2 / 3) * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    tl.store(Y + row_idx * ROW_SIZE + cols, gelu_val, mask=cols < ROW_SIZE)
    tl.store(MEAN + row_idx, mean)
    tl.store(VAR + row_idx, variance)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.numel()
    ROW_SIZE = x.shape[-1]
    output = torch.empty_like(x)
    mean = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    var = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    grid = (x.shape[0],)

    _kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        var,
        N_ELEMENTS=N_ELEMENTS,
        ROW_SIZE=ROW_SIZE,
        eps=eps,
    )

    return output, mean, var