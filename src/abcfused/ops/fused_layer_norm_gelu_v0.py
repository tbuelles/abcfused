# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_layer_norm_gelu(
    X,  # data ptr
    Y,  # output ptr
    W,  # weight ptr
    B,  # bias ptr
    N_ROWS,
    N_COLS,
    MEAN,
    RVAR,
    eps,
    # Block sizes
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols_offsets = tl.arange(0, BLOCK_COLS)
    mask = cols_offsets < N_COLS
    # load data to SRAM
    x = tl.load(X + row_idx * N_COLS + cols_offsets, mask=mask)
    # compute mean
    mean = tl.sum(x, axis=0) / N_COLS
    # compute variance
    var = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    # layer norm
    rvar = 1 / tl.sqrt(var + eps)
    x_norm = (x - mean) * rvar
    # load weight and bias
    weight = tl.load(W + cols_offsets, mask=mask)
    bias = tl.load(B + cols_offsets, mask=mask)
    # apply weight and bias
    x_norm = x_norm * weight + bias
    # gelu
    output = 0.5 * x_norm * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (x_norm + 0.044715 * x_norm**3)))
    # save output to DRAM
    tl.store(Y + row_idx * N_COLS + cols_offsets, output, mask=mask)
    tl.store(MEAN + row_idx, mean)
    tl.store(RVAR + row_idx, rvar)

def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    output = torch.empty_like(x)
    mean = torch.empty(N_ROWS, device=x.device, dtype=x.dtype)
    rvar = torch.empty(N_ROWS, device=x.device, dtype=x.dtype)
    grid = (N_ROWS,)
    _kernel_layer_norm_gelu[grid](
        x,
        output,
        weight,
        bias,
        N_ROWS,
        N_COLS,
        mean,
        rvar,
        eps,
        BLOCK_ROWS=1,
        BLOCK_COLS=1024,
    )
    return output, mean, rvar

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
  return layer_norm_gelu(x, weight, bias, eps)