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
    MEAN,  # pointer to the mean
    RVAR,  # pointer to the variance
    N_ELEMENTS,  # number of elements in the vector
    N_FEATURES,  # number of features in the vector
    eps,  # regularization factor
    **meta
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, meta['BLOCK_SIZE'])
    mask = cols < N_FEATURES

    # Load data to SRAM
    x = tl.load(X + row_idx * N_FEATURES + cols, mask=mask)
    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)

    # Compute mean
    mean = tl.sum(x, axis=0) / N_FEATURES
    # Compute variance
    var = tl.sum((x - mean) * (x - mean), axis=0) / N_FEATURES

    # Normalize
    rvar = 1 / tl.sqrt(var + eps)
    x_hat = (x - mean) * rvar

    # Scale and shift
    x_hat = x_hat * w + b

    # Apply GELU
    output = 0.5 * x_hat * (1 + tl.tanh(0.7978845608028654 * (x_hat + 0.044715 * x_hat * x_hat * x_hat)))

    # Write back the output
    tl.store(Y + row_idx * N_FEATURES + cols, output, mask=mask)
    tl.store(MEAN + row_idx, mean)
    tl.store(RVAR + row_idx, rvar)


def layer_norm_gelu(x, weight, bias, eps=1e-5):
    N_ELEMENTS, N_FEATURES = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N_FEATURES)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8
    num_stages = 3 if BLOCK_SIZE > 2048 else 2

    mean = torch.zeros(N_ELEMENTS, device=x.device, dtype=torch.float32)
    rvar = torch.zeros(N_ELEMENTS, device=x.device, dtype=torch.float32)
    output = torch.empty_like(x)

    _layer_norm_gelu_fwd_kernel[N_ELEMENTS](
        x,
        output,
        weight,
        bias,
        mean,
        rvar,
        N_ELEMENTS,
        N_FEATURES,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return output