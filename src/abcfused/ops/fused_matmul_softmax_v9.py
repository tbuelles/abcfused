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
    SOFTMAX_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    ram = tl.max_contiguous(tl.min(rm, M) - rm, BLOCK_SIZE_M)
    ran = tl.max_contiguous(tl.min(rn, N) - rn, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        rak = tl.max_contiguous(tl.min(rk, K) - rk, BLOCK_SIZE_K)

        a = tl.load(A + (rm[:, None] * K + rk[None, :]), mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(B + (rk[:, None] * N + rn[None, :]), mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    acc = tl.where((rm[:, None] < M) & (rn[None, :] < N), acc, 0.0)

    # softmax over N
    max_val = tl.max(acc, axis=1)
    max_val = tl.where(rm < M, max_val, 0.0)
    max_val = tl.broadcast_to(max_val[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    tmp = tl.exp(acc - max_val)
    tmp = tl.where((rm[:, None] < M) & (rn[None, :] < N), tmp, 0.0)
    sum_exp = tl.sum(tmp, axis=1)
    sum_exp = tl.where(rm < M, sum_exp, 1.0)
    sum_exp = tl.broadcast_to(sum_exp[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    softmax_output = tmp / sum_exp

    tl.store(C + (rm[:, None] * N + rn[None, :]), softmax_output, mask=(rm[:, None] < M) & (rn[None, :] < N))


def fused_matmul_softmax(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    SOFTMAX_N = N

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel_matmul_softmax[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SOFTMAX_N=SOFTMAX_N,
        num_warps=4,
        num_stages=2
    )
    return c