# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C,  # Pointers to matrices
    mean, var,
    W, bias,
    M, N, K,  # Matrix dimensions
    eps,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bm

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c, mask=mask)

    # Layer Norm
    c = tl.where(mask, c, 0.0)

    _mean = tl.sum(c, axis=1) / N
    _var = tl.sum((c - _mean[:, None])**2, axis=1) / N

    tl.store(mean + offs_cm, _mean, mask=offs_cm < M)
    tl.store(var + offs_cm, _var, mask=offs_cm < M)

    c = (c - _mean[:, None]) / tl.sqrt(_var[:, None] + eps)
    c = c * tl.load(W + offs_cn) + tl.load(bias + offs_cn)

    tl.store(C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c, mask=mask)


def fused_matmul_layer_norm(A, B, W, bias, eps):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    mean = torch.empty((M,), device=A.device, dtype=torch.float32)
    var = torch.empty((M,), device=A.device, dtype=torch.float32)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )

    _kernel_matmul_layer_norm[grid](
        A, B, C, mean, var, W, bias,
        M, N, K, eps,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C