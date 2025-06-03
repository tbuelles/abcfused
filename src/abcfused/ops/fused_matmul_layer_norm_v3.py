# type: ignore

import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C, D,  # Pointers to matrices
    M, N, K,  # Matrix dimensions
    stride_am, stride_ak,  # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B
    stride_cm, stride_cn,  # Strides for matrix C
    stride_dm, stride_dn,  # Strides for matrix D (Layer Norm)
    eps,  # LayerNorm epsilon
    mean_ptr, var_ptr,  # Pointers to mean and variance
    W_norm, B_norm, #LayerNorm weight and bias
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.num_programs(axis=1)
    num_pid_m = tl.num_programs(axis=0)

    block_start_m = pid * BLOCK_SIZE_M
    block_start_n = (pid // num_pid_m) * BLOCK_SIZE_N
    # Initialize accumulators
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load block of A and B matrices and compute dot product
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A + (block_start_m * stride_am + k * stride_ak), mask=(block_start_m[:, None] + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(B + (k * stride_bk + block_start_n * stride_bn), mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < K) & ((block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    c = acc.to(tl.float32)
    tl.store(C + (block_start_m * stride_cm + block_start_n * stride_cn), c, mask=(block_start_m[:, None] + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & ((block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < N))
    
    # Layer Norm
    x = tl.load(C + (block_start_m * stride_cm + block_start_n * stride_cn), mask=(block_start_m[:, None] + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & ((block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < N)).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    var = tl.sum((x - mean[:,None])**2, axis=1) / N
    x_norm = (x - mean[:,None]) / tl.sqrt(var[:,None] + eps)
    output = x_norm * W_norm + B_norm
    tl.store(D + (block_start_m * stride_dm + block_start_n * stride_dn), output, mask=(block_start_m[:, None] + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & ((block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < N))


def matmul_layer_norm(a, b, w_norm, b_norm, eps):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    d = torch.empty((M, N), device=a.device, dtype=torch.float32)
    mean = torch.empty((M,), device=a.device, dtype=torch.float32)
    var = torch.empty((M,), device=a.device, dtype=torch.float32)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    grid = (triton.cdiv(M, BLOCK_SIZE_M), )
    _kernel_matmul_layer_norm[grid](
        a, b, c, d,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        d.stride(0), d.stride(1),
        eps,
        mean, var,
        w_norm, b_norm,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return d

fused_matmul_layer_norm = matmul_layer_norm