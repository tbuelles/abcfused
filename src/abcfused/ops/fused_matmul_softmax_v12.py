# type: ignore

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _kernel(
    A_ptr, B_ptr, C_ptr, D_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offset_m = tl.arange(0, BLOCK_SIZE_M)
    offset_n = tl.arange(0, BLOCK_SIZE_N)
    mask_m = (block_start_m + offset_m) < M
    mask_n = (block_start_n + offset_n) < N

    A = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    B = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_block_ptr = A_ptr + (block_start_m + offset_m)[:, None] * K + (k + tl.arange(0, BLOCK_SIZE_K))[None, :]
        b_block_ptr = B_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * N + (block_start_n + offset_n)[None, :]
        a = tl.load(a_block_ptr, mask=(mask_m[:, None] & (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < K), other=0.0)
        b = tl.load(b_block_ptr, mask=((k + tl.arange(0, BLOCK_SIZE_K)[:, None]) < K & mask_n[None, :]), other=0.0)

        accumulator += tl.dot(a, b)

    logits = accumulator
    max_logits = tl.max(logits, axis=1)
    logits = logits - max_logits[:, None]
    exp_logits = tl.exp(logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1)
    softmax_output = exp_logits / sum_exp_logits[:, None]

    d_block_ptr = D_ptr + (block_start_m + offset_m)[:, None] * N + (block_start_n + offset_n)[None, :]
    tl.store(d_block_ptr, softmax_output, mask=mask_m[:, None] & mask_n[None, :])


def fused_matmul_softmax(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.matmul(a, b)
    d = torch.softmax(c, dim=-1)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    num_warps = 4
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    d_triton = torch.empty(M, N, device=a.device, dtype=torch.float32)
    _kernel[grid](
        a, b, c, d_triton,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION="softmax",
        num_warps=num_warps,
    )
    return d_triton