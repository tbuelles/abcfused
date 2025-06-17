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
    N,  # number of rows in X
    M,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < M

    # Compute mean
    x = tl.load(X + row_idx * M + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / M

    # Compute variance
    var = tl.sum((x - mean) ** 2, axis=0) / M

    # Normalize
    x_hat = (x - mean) / tl.sqrt(var + eps)

    # Weight and bias
    weight = tl.load(W + cols, mask=mask, other=1.0)
    bias = tl.load(B + cols, mask=mask, other=0.0)
    x_hat = x_hat * weight + bias

    # GELU
    output = 0.5 * x_hat * (1 + tl.tanh(0.7978845608028654 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))

    # Write back the output
    tl.store(Y + row_idx * M + cols, output, mask=mask)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    output = torch.empty_like(x)
    grid = (N,)
    _layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        N,
        M,
        eps,
        BLOCK_SIZE=M,
    )
    return output