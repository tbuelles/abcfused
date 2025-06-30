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
    ACCUM_TYPE: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    IS_EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    # re-order program ID for better L2 performance
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # do matrix multiplication
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    A = A + rm[:, None] * K + rk[None, :]
    B = B + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_TYPE)

    for k in range(0, K, BLOCK_SIZE_K):
        if not IS_EVEN_K:
            k_mask = (k + rk) < K
            a = tl.load(A + k * BLOCK_SIZE_K, mask=k_mask[None, :], other=0.0)
            b = tl.load(B + k * BLOCK_SIZE_K, mask=k_mask[:, None], other=0.0)
        else:
            a = tl.load(A + k * BLOCK_SIZE_K)
            b = tl.load(B + k * BLOCK_SIZE_K)
        acc += tl.dot(a, b, allow_tf32=False)

    acc = acc.to(tl.float32)

    # apply softmax row-wise
    max_val = tl.max(acc, axis=1, keepdims=True)
    exp_values = tl.exp(acc - max_val)
    sum_exp = tl.sum(exp_values, axis=1, keepdims=True)
    output = exp_values / sum_exp

    C = C + rm[:, None] * N + rn[None, :]
    if not IS_EVEN_M or not IS_EVEN_N:
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(C, output, mask=mask)
    else:
        tl.store(C, output)

def fused_matmul_softmax(a, b):
    M, K = a.shape
    K, N = b.shape
    assert a.is_contiguous()
    assert b.is_contiguous()

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    ACCUM_TYPE = tl.float32

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    _kernel_matmul_softmax[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACCUM_TYPE=ACCUM_TYPE,
        IS_EVEN_M=(M % BLOCK_SIZE_M == 0),
        IS_EVEN_N=(N % BLOCK_SIZE_N == 0),
        IS_EVEN_K=(K % BLOCK_SIZE_K == 0),
    )

    return c