# type: ignore

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_fwd_kernel(
    X, W, B, Mean, Variance, Output,
    N_ELEMENTS: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    BLOCK_SIZE = 1024
    offset = pid * BLOCK_SIZE
    x = X + offset
    w = W + offset
    b = B + offset
    mean = Mean + pid
    variance = Variance + pid

    # Load data
    x_block = tl.load(x, mask=tl.arange(0, BLOCK_SIZE) < N_ELEMENTS - offset, other=0.0)

    # Compute mean and variance
    _mean = tl.sum(x_block, axis=0) / N_ELEMENTS
    _variance = tl.sum((x_block - _mean)**2, axis=0) / N_ELEMENTS

    tl.store(mean, _mean)
    tl.store(variance, _variance)

    # Normalize
    x_norm = (x_block - _mean) / tl.sqrt(_variance + eps)

    # Apply linear transformation
    x_norm = x_norm * tl.load(w, mask=tl.arange(0, BLOCK_SIZE) < N_ELEMENTS - offset, other=0.0) + tl.load(b, mask=tl.arange(0, BLOCK_SIZE) < N_ELEMENTS - offset, other=0.0)

    # Apply GELU activation
    output = 0.5 * x_norm * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))

    # Store output
    tl.store(Output + offset, output, mask=tl.arange(0, BLOCK_SIZE) < N_ELEMENTS - offset)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS = x.shape[-1]
    grid = (N_ELEMENTS // 1024 + (N_ELEMENTS % 1024 != 0),)

    mean = torch.empty((grid[0],), device=x.device, dtype=x.dtype)
    variance = torch.empty((grid[0],), device=x.device, dtype=x.dtype)
    output = torch.empty_like(x)

    _layer_norm_gelu_fwd_kernel[grid](
        x, weight, bias, mean, variance, output,
        N_ELEMENTS=N_ELEMENTS,
        eps=eps
    )
    return output

fused_layer_norm_gelu = layer_norm_gelu