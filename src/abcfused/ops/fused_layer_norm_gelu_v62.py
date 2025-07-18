# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    X, Y, W, B,  # Pointers to data
    N, M,  # shape of the input
    eps,  # layer norm epsilon
    # layer norm output
    mean_ptr,
    var_ptr,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    CACHELINE_SIZE_M: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    row_start = row_idx * BLOCK_SIZE_N
    col_start = col_idx * BLOCK_SIZE_M

    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < M

    x_ptrs = X + row_start * M + offsets_m
    x = tl.load(x_ptrs, mask=mask_m, other=0.0)

    # Compute mean and variance
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / M

    sum_x2 = tl.sum(x * x, axis=0)
    var = sum_x2 / M - mean * mean

    # Normalize
    invstd = 1.0 / tl.sqrt(var + eps)
    x_hat = (x - mean) * invstd

    # Linear transform
    w_ptrs = W + offsets_m
    w = tl.load(w_ptrs, mask=mask_m, other=1.0)
    b_ptrs = B + offsets_m
    b = tl.load(b_ptrs, mask=mask_m, other=0.0)

    x_transformed = x_hat * w + b

    # Apply GELU
    sqrt_2_over_pi = 0.7978845608028654
    gelu_val = 0.5 * x_transformed * (1.0 + tl.tanh(sqrt_2_over_pi * x_transformed * (1.0 + 0.044715 * x_transformed * x_transformed)))

    # Write back the output
    y_ptrs = Y + row_start * M + offsets_m
    tl.store(y_ptrs, gelu_val, mask=mask_m)

    # Write back mean and variance for verification
    tl.store(mean_ptr + row_start, mean)
    tl.store(var_ptr + row_start, var)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device='cuda')
    var = torch.empty((N,), dtype=torch.float32, device='cuda')

    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128
    CACHELINE_SIZE_M = 32
    grid = (N // BLOCK_SIZE_N , triton.cdiv(M, BLOCK_SIZE_M))

    _kernel[grid](
        x,
        y,
        weight,
        bias,
        N,
        M,
        eps,
        mean,
        var,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        CACHELINE_SIZE_M=CACHELINE_SIZE_M
    )
    return y