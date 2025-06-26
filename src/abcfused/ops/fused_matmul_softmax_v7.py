# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_softmax(
    A, B, C,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # re-arrange block id for better memory access
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # do matrix multiplication
    # m_range = tl.arange(0, BLOCK_SIZE_M)
    # n_range = tl.arange(0, BLOCK_SIZE_N)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_start = k
        k_end = min(k + BLOCK_SIZE_K, K)

        offs_ak = k_start + tl.arange(0, BLOCK_SIZE_K)
        offs_bk = k_start + tl.arange(0, BLOCK_SIZE_K)

        a = tl.load(A + (offs_am[:, None] * K + offs_ak[None, :]), mask=(offs_am[:, None] < M) & (offs_ak[None, :] < K), other=0.0)
        b = tl.load(B + (offs_bk[:, None] * N + offs_bn[None, :]), mask=(offs_bk[:, None] < K) & (offs_bn[None, :] < N), other=0.0)

        accumulator += tl.dot(a, tl.trans(b))

    accumulator = tl.where((offs_am[:, None] < M) & (offs_bn[None, :] < N), accumulator, -float('inf'))

    # softmax
    max_val = tl.max(accumulator, axis=1)
    accumulator = accumulator - max_val[:, None]
    exp_val = tl.exp(accumulator)
    sum_exp = tl.sum(exp_val, axis=1)
    softmax_output = exp_val / sum_exp[:, None]

    # store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(C + offs_cm[:, None] * N + offs_cn[None, :], softmax_output, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def fused_matmul_softmax(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    _kernel_matmul_softmax[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,
        num_stages=2
    )
    return c