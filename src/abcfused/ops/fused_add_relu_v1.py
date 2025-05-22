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
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + row_idx * x_stride + cols
    y_ptrs = y_ptr + row_idx * y_stride + cols
    mask = cols < N
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    output = x + y
    output = tl.where(output > 0, output, 0.0)
    output_ptrs = output_ptr + row_idx * output_stride + cols
    tl.store(output_ptrs, output, mask=mask)

def fused_add_relu(x: torch.Tensor, y: torch.Tensor):
    N = x.shape[1]
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (x.shape[0],)

    _kernel[grid](
        x,
        y,
        output,
        x.stride(0),
        y.stride(0),
        output.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output