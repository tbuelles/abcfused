# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weight
    B,  # pointer to the bias
    N_ELEMENTS: tl.constexpr,  # number of elements in the vector
    eps: tl.constexpr,  # epsilon to avoid division by zero
    act_kind: tl.constexpr,  # gelu approximation (0=tanh, 1=fast_gelu)
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = row_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    x = tl.load(X + offsets, mask=mask, other=0.0)

    # LayerNorm
    mean = tl.sum(x, axis=0) / N_ELEMENTS
    variance = tl.sum((x - mean) ** 2, axis=0) / N_ELEMENTS
    norm = (x - mean) / tl.sqrt(variance + eps)

    # Scale and shift
    norm = norm * tl.load(W) + tl.load(B)

    # GELU
    if act_kind == 0:
        gelu = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (norm + 0.044715 * norm * norm * norm)))
    else:
        gelu = norm * 0.5 * (1.0 + tl.erf(norm / 1.4142135623730951))

    tl.store(Y + offsets, gelu, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5, act_kind=0):
    """
    Fused LayerNorm + GELU.
    """
    N_ELEMENTS = x.shape[-1]
    output = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    if BLOCK_SIZE > 2048:
        BLOCK_SIZE = 2048
    grid = (x.shape[0],)

    _kernel[grid](
        x,
        output,
        weight,
        bias,
        N_ELEMENTS=N_ELEMENTS,
        eps=eps,
        act_kind=act_kind,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output