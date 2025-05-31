# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_softmax(
    A, B, C, M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_a = offs_am[:, None] + k + tl.arange(0, BLOCK_SIZE_K)[None, :]
        offs_b = (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) + offs_bn[None, :]

        a = tl.load(A + offs_a * K, mask=offs_a < M[:, None], other=0.0)
        b = tl.load(B + offs_b * N, mask=offs_b < K[:, None], other=0.0)

        acc += tl.dot(a, b)

    max_val = tl.max(acc, axis=1)
    max_val = tl.where(max_val != 0, max_val, 1.0)
    exp_values = tl.exp(acc - max_val[:, None])
    sum_exp = tl.sum(exp_values, axis=1)
    softmax_output = exp_values / sum_exp[:, None]


    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C + offs_cm[:, None] * N + offs_cn[None, :], softmax_output, mask=mask)

def fused_matmul_softmax(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    _kernel_matmul_softmax[grid](
        a, b, c, M, N, K,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=2
    )
    return c