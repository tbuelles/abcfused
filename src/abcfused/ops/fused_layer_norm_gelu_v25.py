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
    norm_var, # pointer to the intermediate layer norm variance
    norm_mean, # pointer to the intermediate layer norm mean
    N,  # number of rows in X
    D,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # Load data
    x = tl.load(X + row_idx * D + cols, mask=mask, other=0.0)

    # LayerNorm
    mean = tl.sum(x, axis=0) / D
    variance = tl.sum((x - mean)**2, axis=0) / D

    x_hat = (x - mean) / tl.sqrt(variance + eps)
    
    # Save mean and variance
    tl.store(norm_mean + row_idx, mean)
    tl.store(norm_var + row_idx, variance)

    # Scale and bias
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    x_scaled = x_hat * w + b

    # GELU
    gelu_val = 0.5 * x_scaled * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.math.pi) * (x_scaled + 0.044715 * x_scaled**3)))

    # Store the result
    tl.store(Y + row_idx * D + cols, gelu_val, mask=mask)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, D = x.shape
    output = torch.empty_like(x)
    norm_var = torch.empty((N,), dtype=torch.float32, device=x.device)
    norm_mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    grid = (N,)

    _layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        norm_var,
        norm_mean,
        N,
        D,
        eps,
        BLOCK_SIZE=D,
    )
    return output, norm_mean, norm_var