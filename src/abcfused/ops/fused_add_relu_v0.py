# type: ignore

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    added = x + y
    relu_output = tl.where(added > 0, added, 0)

    tl.store(output_ptr + offsets, relu_output, mask=mask)


def fused_add_relu(x: torch.Tensor, y: torch.Tensor):
    assert x.is_contiguous() and y.is_contiguous(), "Inputs must be contiguous"
    n_elements = x.numel()
    output = torch.empty_like(x)

    GRID = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _kernel[GRID](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return output