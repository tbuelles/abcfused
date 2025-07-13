# type: ignore

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C,  # pointers to matrices
    mean, var,
    W, b,
    M, N, K,  # matrix dimensions
    eps,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.num_programs(axis=1)
    num_pid_m = tl.num_programs(axis=0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    A = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B = B + rk[:, None] * stride_bk + rn[None, :] * stride_bm
    C = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    wm = (rm < M)[:, None]
    wn = (rn < N)[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + k * stride_ak, mask=wm & (k + rk[None, :] < K), other=0.0)
        b = tl.load(B + k * stride_bk, mask=wn & (k + rk[:, None] < K), other=0.0)
        acc += tl.dot(a, b)

    acc = acc.to(tl.float32)
    tl.store(C, acc, mask=wm & wn)

    # Layer norm
    C_block = tl.load(C, mask=wm & wn, other=0.0)

    mean_output = tl.sum(C_block, axis=1) / N
    var_output = tl.sum((C_block - mean_output[:, None]) ** 2, axis=1) / N
    tl.store(mean + rm, mean_output, mask=rm < M)
    tl.store(var + rm, var_output, mask=rm < M)

    C_norm = (C_block - mean_output[:, None]) / tl.sqrt(var_output[:, None] + eps)
    C_norm = C_norm * tl.load(W + rn, mask=wn, other=1.0) + tl.load(b + rn, mask=wn, other=0.0)

    tl.store(C, C_norm, mask=wm & wn)

def fused_matmul_layer_norm(A, B, W, b, eps):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    mean = torch.empty((M,), device=A.device, dtype=A.dtype)
    var = torch.empty((M,), device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    _kernel_matmul_layer_norm[grid](
        A, B, C,
        mean, var,
        W, b,
        M, N, K,
        eps,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return C