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
    SOFTMAX_NORM: tl.constexpr,
    ROW_STRIDE_A, COL_STRIDE_A,
    ROW_STRIDE_B, COL_STRIDE_B,
    ROW_STRIDE_C, COL_STRIDE_C,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_am[:, None] * ROW_STRIDE_A + offs_k[None, :] * COL_STRIDE_A
    b_ptrs = B + offs_k[:, None] * ROW_STRIDE_B + offs_bn[None, :] * COL_STRIDE_B

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask= (offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask= (offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * COL_STRIDE_A
        b_ptrs += BLOCK_SIZE_K * ROW_STRIDE_B

    accumulator = accumulator.to(tl.float32)
    max_val = tl.max(accumulator, axis=1)
    logits_exp = tl.exp(accumulator - max_val[:, None])
    sum_exp = tl.sum(logits_exp, axis=1)
    if SOFTMAX_NORM:
        output = logits_exp / sum_exp[:, None]
    else:
        output = logits_exp

    c_ptrs = C + offs_am[:, None] * ROW_STRIDE_C + offs_bn[None, :] * COL_STRIDE_C
    tl.store(c_ptrs, output.to(tl.float32), mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def fused_matmul_softmax(A, B, softmax_norm=True):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    _kernel_matmul_softmax[grid](
        A, B, C,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SOFTMAX_NORM = softmax_norm,
        ROW_STRIDE_A=A.stride(0), COL_STRIDE_A=A.stride(1),
        ROW_STRIDE_B=B.stride(0), COL_STRIDE_B=B.stride(1),
        ROW_STRIDE_C=C.stride(0), COL_STRIDE_C=C.stride(1),
        num_warps=4,
        num_stages=2,
    )
    return C