# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _layer_norm_gelu_fwd_kernel(
    X,  # pointer to the input tensor
    Y,  # pointer to the output tensor
    W,  # pointer to the weight tensor
    B,  # pointer to the bias tensor
    NORM_MEAN,
    NORM_VAR,
    N_COL,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COL
    x = tl.load(X + row_idx * N_COL + cols, mask=mask, other=0.0)

    # Layer Norm
    mean = tl.load(NORM_MEAN + row_idx * 1).to(tl.float32)
    var = tl.load(NORM_VAR + row_idx * 1).to(tl.float32)
    x_hat = (x - mean) / tl.sqrt(var + eps)
    weight = tl.load(W + cols, mask=mask, other=1.0)
    bias = tl.load(B + cols, mask=mask, other=0.0)
    x_normed = x_hat * weight + bias

    # GELU
    gelu_out = 0.5 * x_normed * (1.0 + tl.tanh(0.7978845608028654 * (x_normed + 0.044715 * x_normed * x_normed * x_normed)))

    tl.store(Y + row_idx * N_COL + cols, gelu_out, mask=mask)

def fused_layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ROW, N_COL = x.shape
    mean = torch.empty((N_ROW,), device=x.device, dtype=torch.float32)
    var = torch.empty((N_ROW,), device=x.device, dtype=torch.float32)

    x_normalized = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    torch.layer_norm(x, normalized_shape=[N_COL], weight=weight, bias=bias, eps=eps, out=x_normalized)

    output = torch.empty(x.shape, device=x.device, dtype=torch.float32)

    _layer_norm_gelu_fwd_kernel[(N_ROW,)](
        x,
        output,
        weight,
        bias,
        mean,
        var,
        N_COL,
        eps,
        BLOCK_SIZE=N_COL,
    )
    return output