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
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, N_ELEMENTS)
    X_ptr = X + row_idx * N_ELEMENTS + cols
    W_ptr = W + cols
    B_ptr = B + cols

    x = tl.load(X_ptr, mask=cols < N_ELEMENTS, other=0.0)

    mean = tl.sum(x, axis=0) / N_ELEMENTS
    variance = tl.sum((x - mean) * (x - mean), axis=0) / N_ELEMENTS
    x_norm = (x - mean) / tl.sqrt(variance + eps)

    w = tl.load(W_ptr, mask=cols < N_ELEMENTS, other=1.0)
    b = tl.load(B_ptr, mask=cols < N_ELEMENTS, other=0.0)

    x_norm = x_norm * w + b
    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    Y_ptr = Y + row_idx * N_ELEMENTS + cols
    tl.store(Y_ptr, gelu, mask=cols < N_ELEMENTS)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    output = torch.empty_like(x)
    N_ELEMENTS = x.shape[-1]
    grid = (x.shape[0],)

    _layer_norm_gelu_kernel[grid](
        x, weight, bias,
        output,
        N_ELEMENTS=N_ELEMENTS,
        eps=eps
    )

    return output