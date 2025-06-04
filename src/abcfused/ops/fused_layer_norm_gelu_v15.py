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
    RVAR,
    N_ELEMENTS,
    eps,
    gelu_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    x = tl.load(X + offsets, mask=mask)

    # LayerNorm
    mean = tl.sum(x, axis=0) / N_ELEMENTS
    var = tl.sum((x - mean) * (x - mean), axis=0) / N_ELEMENTS
    rvar = 1 / tl.sqrt(var + eps)
    x_norm = (x - mean) * rvar

    if W is not None and B is not None:
        w = tl.load(W + offsets, mask=mask)
        b = tl.load(B + offsets, mask=mask)
        x_norm = x_norm * w + b
    tl.store(MEAN + pid, mean)
    tl.store(RVAR + pid, rvar)

    # GELU approximation
    if gelu_dtype == "approx":
        gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
    elif gelu_dtype == "exact":
        gelu = x_norm * 0.5 * (1.0 + tl.erf(x_norm / 1.41421356237))
    else:
        gelu = x_norm
    tl.store(Y + offsets, gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5, gelu_dtype="approx"):
    shape = x.shape
    dtype = x.dtype
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    y = torch.empty_like(x)
    mean = torch.empty((num_blocks,), dtype=dtype, device=x.device)
    rvar = torch.empty((num_blocks,), dtype=dtype, device=x.device)

    _kernel[(num_blocks,)](
        y,
        x,
        weight,
        bias,
        mean,
        rvar,
        n_elements,
        eps,
        gelu_dtype=gelu_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y, mean, rvar