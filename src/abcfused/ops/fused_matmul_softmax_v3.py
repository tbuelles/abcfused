# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_softmax(
    A, B, C,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = B + (offs_k[:, None] * N + offs_bn[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * K
        b_ptrs += BLOCK_SIZE_K * N

    accumulator = tl.where(offs_am[:, None] < M, accumulator, -float('inf'))
    accumulator = tl.where(offs_bn[None, :] < N, accumulator, -float('inf'))

    max_val = tl.max(accumulator, axis=1)
    max_val = max_val[:, None]
    accumulator = tl.exp(accumulator - max_val)

    row_sum = tl.sum(accumulator, axis=1)
    row_sum = row_sum[:, None]
    accumulator = accumulator / row_sum

    c_ptrs = C + (offs_am[:, None] * N + offs_bn[None, :])
    tl.store(c_ptrs, accumulator, mask= (offs_am[:, None] < M) & (offs_bn[None, :] < N))


def fused_matmul_softmax(a, b):
    M, K = a.shape
    K, N = b.shape
    assert a.is_contiguous()
    assert b.is_contiguous()
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    _kernel_matmul_softmax[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
        num_stages=2,
    )
    return c