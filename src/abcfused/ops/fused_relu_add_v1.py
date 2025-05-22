# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    x_stride,
    y_stride,
    output_stride,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets * x_stride, mask=mask)
    y = tl.load(y_ptr + offsets * y_stride, mask=mask)

    relu_x = tl.where(x > 0, x, 0)
    output = relu_x + y

    tl.store(output_ptr + offsets * output_stride, output, mask=mask)


def fused_relu_add(x: torch.Tensor, y: torch.Tensor):
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _kernel[grid](
        x,
        y,
        output,
        x.stride(0),
        y.stride(0),
        output.stride(0),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output