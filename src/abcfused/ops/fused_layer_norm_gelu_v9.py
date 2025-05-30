# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Var,  # pointer to the variance
    N_ROWS,  # number of rows in X
    N_COLS,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    # BLOCK_SIZE: it's a hint so the compiler can pick a good value
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    # load data
    x = tl.load(X + row_idx * N_COLS + col_offsets, mask=mask)

    # calculate mean
    mean = tl.sum(x, axis=0) / N_COLS
    tl.store(Mean + row_idx, mean)

    # calculate variance
    var = tl.sum((x - mean) ** 2, axis=0) / N_COLS
    tl.store(Var + row_idx, var)

    # normalize
    x_hat = (x - mean) / tl.sqrt(var + eps)

    # apply weights and biases
    w = tl.load(W + col_offsets, mask=mask)
    b = tl.load(B + col_offsets, mask=mask)
    x_hat = x_hat * w + b

    # gelu
    output = x_hat * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))

    # write back to Y
    tl.store(Y + row_idx * N_COLS + col_offsets, output, mask=mask)

def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROWS, N_COLS = x.shape
    Mean = torch.empty((N_ROWS,), device=x.device, dtype=torch.float32)
    Var = torch.empty((N_ROWS,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    grid = (N_ROWS,)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        Mean,
        Var,
        N_ROWS,
        N_COLS,
        eps,
        BLOCK_SIZE=1024,
    )
    return output