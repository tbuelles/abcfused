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
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    # X = tl.load(X + row_idx * M + offsets, mask=mask, other=0.).to(tl.float32)
    x = tl.load(X + row_idx * M + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.).to(tl.float32)

    mean = tl.sum(x, axis=0) / M
    var = tl.sum((x - mean) * (x - mean), axis=0) / M
    x_norm = (x - mean) / tl.sqrt(var + eps)
    
    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.).to(tl.float32)
    b = tl.load(B + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.).to(tl.float32)
    
    output = x_norm * w + b
    gelu_output = output * 0.5 * (1.0 + tl.tanh(0.7978845608 * (output + 0.044715 * output * output * output)))
    
    tl.store(Y + row_idx * M + tl.arange(0, BLOCK_SIZE), gelu_output, mask=mask)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N, M = x.shape
    output = torch.empty_like(x)
    grid = (N,)
    _layer_norm_gelu_kernel[grid](
        x, weight, bias, output,
        N, M,
        eps,
        BLOCK_SIZE=M,
        num_warps=4,
    )
    return output