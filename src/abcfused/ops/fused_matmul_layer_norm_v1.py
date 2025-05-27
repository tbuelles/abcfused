# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C,  # pointers to matrices
    mean, variance,  # pointers to layer norm stats
    W, b,  # pointers to layer norm weights and biases
    M, N, K,  # matrix dimensions
    eps,  # layer norm epsilon
    stride_am, stride_ak,  # strides for A
    stride_bk, stride_bn,  # strides for B
    stride_cm, stride_cn,  # strides for C
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask= (offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask= (offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn

    # Layer Norm
    mean_block = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    var_block = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for m in range(BLOCK_SIZE_M):
      if offs_am[m] < M:
          mean_block[m] = tl.sum(c[m,:], axis=0)
          var_block[m] = tl.sum(c[m,:] * c[m,:], axis=0)

    tl.store(mean + offs_am, mean_block, mask=offs_am < M)
    tl.store(variance + offs_am, var_block, mask=offs_am < M)

    mean_val = tl.load(mean + offs_am, mask=offs_am < M) / N
    var_val = tl.load(variance + offs_am, mask=offs_am < M) / N - mean_val * mean_val
    invstd = 1 / tl.sqrt(var_val + eps)

    w = tl.load(W + offs_am, mask=offs_am < M)
    b_val = tl.load(b + offs_am, mask=offs_am < M)
    output = (c - mean_val[:, None]) * invstd[:, None] * w[:, None] + b_val[:, None]

    tl.store(c_ptrs, output, mask= (offs_am[:, None] < M) & (offs_cn[None, :] < N))

def fused_matmul_layer_norm(A, B, W, b, eps):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    mean = torch.empty((M,), device=A.device, dtype=A.dtype)
    variance = torch.empty((M,), device=A.device, dtype=A.dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    _kernel_matmul_layer_norm[grid](
        A, B, C,
        mean, variance,
        W, b,
        M, N, K,
        eps,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C