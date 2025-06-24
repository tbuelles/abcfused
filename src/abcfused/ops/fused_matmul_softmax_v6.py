# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_softmax(
    A, B, C, M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACCUM_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # re-order program ID for better L2 performance
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # do matrix multiplication
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_TYPE)

    for kk in range(0, K, BLOCK_SIZE_K):
        k = kk + tl.arange(0, BLOCK_SIZE_K)
        a = tl.load(A + (rm[:, None] * K + k[None, :]), mask=(rm[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(B + (k[:, None] * N + rn[None, :]), mask=(k[:, None] < K) & (rn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)

    C_block = accumulator.to(tl.float32)

    # softmax along rows
    row_max = tl.max(C_block, axis=1)
    C_block = C_block - row_max[:, None]
    C_block_exp = tl.exp(C_block)
    row_sum = tl.sum(C_block_exp, axis=1)
    C_block = C_block_exp / row_sum[:, None]

    # write back result
    cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    tl.store(C + (cm[:, None] * N + cn[None, :]), C_block, mask=(cm[:, None] < M) & (cn[None, :] < N))


def fused_matmul_softmax(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel_matmul_softmax[grid](
        a, b, c, M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACCUM_TYPE=tl.float32,
    )
    return c