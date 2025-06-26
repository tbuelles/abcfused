# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_kernel(
    X, W, B,
    Mean, Var,
    Out,
    N_COL,
    eps,
    **meta
):
    row_idx = tl.program_id(0)
    cols_offsets = tl.arange(0, meta['BLOCK_SIZE'])
    mask = cols_offsets < N_COL

    x = tl.load(X + row_idx * N_COL + cols_offsets, mask=mask)

    mean = tl.load(Mean + row_idx)
    var = tl.load(Var + row_idx)

    x_norm = (x - mean) * tl.rsqrt(var + eps)
    w = tl.load(W + cols_offsets, mask=mask)
    b = tl.load(B + cols_offsets, mask=mask)
    x_norm = x_norm * w + b

    gelu = x_norm * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * x_norm * (1.0 + 0.044715 * x_norm * x_norm)))

    tl.store(Out + row_idx * N_COL + cols_offsets, gelu, mask=mask)

def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROW, N_COL = x.shape
    mean = torch.empty((N_ROW,), device=x.device, dtype=torch.float32)
    var = torch.empty((N_ROW,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    _layer_norm_gelu_kernel[(N_ROW,)](
        x, weight, bias,
        mean, var,
        output,
        N_COL,
        eps,
        BLOCK_SIZE=1024,
        num_warps=4,
        num_stages=1,
    )

    return output