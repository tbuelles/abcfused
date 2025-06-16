# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, bias, gamma, var,
    C,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = B + (offs_k[:, None] * N + offs_bn[None, :])

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * A.stride(1)
        b_ptrs += BLOCK_SIZE_K * B.stride(0)

    accumulator = accumulator.to(C.dtype.element_ty)

    # LayerNorm part
    mean = tl.sum(accumulator, axis=1) / N
    variance = tl.sum((accumulator - mean[:, None])**2, axis=1) / N
    inv_std = 1 / tl.sqrt(variance + var)
    output = (accumulator - mean[:, None]) * (gamma * inv_std)[:, None] + bias[:, None]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * N + offs_cn[None, :]
    tl.store(c_ptrs, output, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def fused_matmul_layer_norm(a, b, bias, gamma, var):
    M, K = a.shape
    K, N = b.shape
    C = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GRID = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel_matmul_layer_norm[GRID](
        a, b, bias, gamma, var,
        C,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
    )
    return C