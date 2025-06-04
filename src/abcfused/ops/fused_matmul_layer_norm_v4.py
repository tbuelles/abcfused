# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C, D,  # Pointers to matrices
    M, N, K,  # Matrix dimensions
    eps,  # layer_norm epsilon
    stride_am, stride_ak,  # Stride for matrix A
    stride_bk, stride_bn,  # Stride for matrix B
    stride_cm, stride_cn,  # Stride for matrix C
    stride_dm, stride_dn,  # Stride for matrix D (LayerNorm output)
    W,  # LayerNorm weight
    B_ln, # LayerNorm bias
    stride_w, # LayerNorm weight stride
    stride_b, # LayerNorm bias stride
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + c_offs_m[:, None] * stride_cm + c_offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(c_offs_m[:, None] < M) & (c_offs_n[None, :] < N))


    # LayerNorm
    c_block = tl.load(c_ptrs, mask=(c_offs_m[:, None] < M) & (c_offs_n[None, :] < N), other=0.0)
    mean = tl.sum(c_block, axis=1) / N
    variance = tl.sum((c_block - mean[:, None])**2, axis=1) / N
    norm = (c_block - mean[:, None]) / tl.sqrt(variance[:, None] + eps)

    weight = tl.load(W + c_offs_n, mask=c_offs_n < N, other=1.0)
    bias = tl.load(B_ln + c_offs_n, mask=c_offs_n < N, other=0.0)
    output = norm * weight[None, :] + bias[None, :]

    d_ptrs = D + c_offs_m[:, None] * stride_dm + c_offs_n[None, :] * stride_dn
    tl.store(d_ptrs, output, mask=(c_offs_m[:, None] < M) & (c_offs_n[None, :] < N))


def fused_matmul_layer_norm(a, b, w, bias, eps):
    M, K = a.shape
    K, N = b.shape
    C = torch.empty((M, N), device=a.device, dtype=a.dtype)
    D = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    _kernel_matmul_layer_norm[grid](
        a, b, C, D,
        M, N, K,
        eps,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        w, bias,
        w.stride(0), bias.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return D