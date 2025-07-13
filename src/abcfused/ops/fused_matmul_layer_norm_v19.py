# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, Mean, Variance, C,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_mean, stride_var,
    M, N, K,
    eps,
    W, B_layernorm, # LayerNorm weights and biases
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (c_offs_m[:, None] < M) & (c_offs_n[None, :] < N)
    C_pre_layernorm = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    C_pre_layernorm = tl.where(mask, accumulator, C_pre_layernorm)

    mean = tl.sum(C_pre_layernorm, axis=1) / N
    variance = tl.sum((C_pre_layernorm - mean[:, None])**2, axis=1) / N
    C_layernorm = (C_pre_layernorm - mean[:, None]) / tl.sqrt(variance[:, None] + eps)
    C_layernorm = C_layernorm * W + B_layernorm

    c_ptrs = C + (c_offs_m[:, None] * stride_cm + c_offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, C_layernorm, mask=mask)

    mean_ptrs = Mean + c_offs_m
    var_ptrs = Variance + c_offs_m

    tl.store(mean_ptrs, mean, mask=c_offs_m < M)
    tl.store(var_ptrs, variance, mask=c_offs_m < M)


def fused_matmul_layer_norm(A, B, W, B_layernorm, eps):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    Mean = torch.empty((M,), device=A.device, dtype=torch.float32)
    Variance = torch.empty((M,), device=A.device, dtype=torch.float32)

    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel_matmul_layer_norm[grid](
        A, B, Mean, Variance, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        Mean.stride(0), Variance.stride(0),
        M, N, K,
        eps,
        W, B_layernorm,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return C, Mean, Variance