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
    NORM_MEAN, # pointer to the intermediate mean
    NORM_VAR, # pointer to the intermediate variance
    N_COL,  # number of columns in X
    N_ROW,  # number of rows in X
    eps,    # layer norm epsilon
    gelu_approx, #gelu approximation
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COL

    # load data to SRAM
    x = tl.load(X + row_idx * N_COL + cols, mask=mask)
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)

    # compute mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / N_COL

    # compute variance
    var = tl.sum((x - mean) * (x - mean), axis=0) / N_COL

    # layer norm
    norm_x = (x - mean) / tl.sqrt(var + eps)
    norm_x = norm_x * w + b

    #GELU
    if gelu_approx == 0:
        gelu_x = 0.5 * norm_x * (1.0 + tl.tanh(tl.sqrt(2.0 / tl.pi) * (norm_x + 0.044715 * tl.pow(norm_x, 3.0))))
    else:
        gelu_x = norm_x * 0.5 * (1.0 + tl.erf(norm_x / tl.sqrt(2.0)))
    # write back to DRAM
    tl.store(Y + row_idx * N_COL + cols, gelu_x, mask=mask)
    tl.store(NORM_MEAN + row_idx, mean)
    tl.store(NORM_VAR + row_idx, var)


def fused_layer_norm_gelu(x, weight, bias, eps=1e-5, gelu_approx=0):
    N_ROW, N_COL = x.shape
    output = torch.empty_like(x)
    norm_mean = torch.empty((N_ROW,), dtype=torch.float32, device=x.device)
    norm_var = torch.empty((N_ROW,), dtype=torch.float32, device=x.device)
    grid = (N_ROW,)
    BLOCK_SIZE = min(triton.next_power_of_2(N_COL), 2048)
    _layer_norm_gelu_kernel[grid](
        x,
        output,
        weight,
        bias,
        norm_mean,
        norm_var,
        N_COL,
        N_ROW,
        eps,
        gelu_approx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output