# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    X, # pointer to the input
    W, # pointer to the output
    N_ELEMENTS: tl.constexpr, # number of elements in the vector
    BLOCK_SIZE: tl.constexpr, # number of elements each program should process
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    x = tl.load(X + offsets, mask=mask)
    # ReLU
    relu_output = tl.where(x > 0, x, 0)
    # Sigmoid
    sigmoid_output = 1 / (1 + tl.exp(-relu_output))
    tl.store(W + offsets, sigmoid_output, mask=mask)

def fused_relu_sigmoid(x):
    output = torch.empty_like(x)
    N_ELEMENTS = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta['BLOCK_SIZE']),)
    _kernel[grid](x, output, N_ELEMENTS=N_ELEMENTS, BLOCK_SIZE=BLOCK_SIZE)
    return output