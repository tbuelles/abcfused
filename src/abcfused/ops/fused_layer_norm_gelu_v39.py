# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B, Y,
    N, M,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < M
    X_row = X + row_idx * M
    W_ptr = W
    B_ptr = B

    x = tl.load(X_row + cols, mask=mask, other=0.0)

    # Calculate mean
    x_mean = tl.sum(x, axis=0) / M

    # Calculate variance
    x_var = tl.sum((x - x_mean)**2, axis=0) / M

    # Normalize
    x_norm = (x - x_mean) / tl.sqrt(x_var + eps)

    # Apply weights and biases
    w = tl.load(W_ptr + cols, mask=mask, other=1.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    x_scaled = x_norm * w + b

    # Apply GELU
    gelu = x_scaled * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x_scaled + 0.044715 * x_scaled * x_scaled * x_scaled)))

    # Write back to memory
    Y_row = Y + row_idx * M
    tl.store(Y_row + cols, gelu, mask=mask)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(M)
    grid = (N,)
    _layer_norm_gelu_kernel[grid](
        x, weight, bias, output,
        N, M,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output