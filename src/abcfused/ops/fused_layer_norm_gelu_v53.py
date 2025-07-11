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
    N_ELEMENTS: tl.constexpr,  # number of elements in the input
    D_MODEL: tl.constexpr,  # dimension of the input
    EPS: tl.constexpr,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,  # size of the block
):
    row_idx = tl.program_id(0)
    row_start = row_idx * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Load data from x
    x = tl.load(X + offsets * D_MODEL, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=1) / D_MODEL
    variance = tl.sum(tl.square(x - mean[:, None]), axis=1) / D_MODEL

    # Save mean and variance
    tl.store(Mean + row_idx, mean, mask=mask)
    tl.store(Var + row_idx, variance, mask=mask)

    # Normalize
    x_hat = (x - mean[:, None]) / tl.sqrt(variance[:, None] + EPS)

    # Scale and bias
    x_hat = x_hat * tl.load(W, mask=mask, other=0.0) + tl.load(B, mask=mask, other=0.0)

    # GELU activation
    gelu_out = x_hat * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))

    # Store the result
    tl.store(Y + offsets * D_MODEL, gelu_out, mask=mask)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS, D_MODEL = x.shape
    BLOCK_SIZE = triton.next_power_of_2(x.shape[1])
    grid = (N_ELEMENTS,)

    mean = torch.empty((N_ELEMENTS,), device=x.device, dtype=torch.float32)
    var = torch.empty((N_ELEMENTS,), device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    _layer_norm_gelu_fwd_kernel[grid](
        x,
        output,
        weight,
        bias,
        mean,
        var,
        N_ELEMENTS=N_ELEMENTS,
        D_MODEL=D_MODEL,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return output