# type: ignore

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _kernel(
    A_ptr, B_ptr, C_ptr,
    mean_ptr, rstd_ptr,
    W_ptr, B_ptr_ln,
    M, N, K,
    eps,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    stride_mean, stride_rstd,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max(rm[:, None], 0)
    ran = tl.max(rn[None, :], 0)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        rk = k + tl.arange(0, BLOCK_SIZE_K)
        A = tl.load(A_ptr + ram[:, None] * stride_am + rk[None, :] * stride_ak, mask=(ram[:, None] < M) & (rk[None, :] < K), other=0.0)
        B = tl.load(B_ptr + rk[:, None] * stride_bk + ran[None, :] * stride_bm, mask=(rk[:, None] < K) & (ran[None, :] < N), other=0.0)
        acc += tl.dot(A, B)
    
    C = acc.to(tl.float32)
    tl.store(C_ptr + ram[:, None] * stride_cm + ran[None, :] * stride_cn, C, mask=(ram[:, None] < M) & (ran[None, :] < N))

    # LayerNorm part
    row_sum = tl.sum(C, axis=1)
    mean = row_sum / N
    tl.store(mean_ptr + rm, mean, mask=rm < M)

    row_variance = tl.sum((C - mean[:, None])**2, axis=1)
    variance = row_variance / N
    rstd = 1 / tl.sqrt(variance + eps)
    tl.store(rstd_ptr + rm, rstd, mask=rm < M)

    C_normalized = (C - mean[:, None]) * rstd[:, None]
    W = tl.load(W_ptr + rn, mask=ran < N, other=0.0)
    B_ln = tl.load(B_ptr_ln + rn, mask=ran < N, other=0.0)
    output = C_normalized * W[None, :] + B_ln[None, :]
    tl.store(C_ptr + ram[:, None] * stride_cm + ran[None, :] * stride_cn, output, mask=(ram[:, None] < M) & (ran[None, :] < N))

def fused_matmul_layer_norm(A, B, W, B_ln, eps):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    mean = torch.empty((M,), device=A.device, dtype=A.dtype)
    rstd = torch.empty((M,), device=A.device, dtype=A.dtype)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    _kernel[grid](
        A, B, C,
        mean, rstd,
        W, B_ln,
        M, N, K,
        eps,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        mean.stride(0), rstd.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C