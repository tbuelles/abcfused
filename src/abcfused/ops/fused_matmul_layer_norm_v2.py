# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_matmul_layer_norm(
    A, B, C, D, E,  # pointers to matrices
    M, N, K,  # dimensions of matrices
    eps,  # layer norm epsilon
    output_row_stride,
    # strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_dm, stride_dn,
    # Other flags
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)
    mean = tl.sum(c, axis=1) / N
    var = tl.sum((c - mean[:, None])**2, axis=1) / N
    c = (c - mean[:, None]) / tl.sqrt(var[:, None] + eps)
    c = c * tl.load(D + offs_am, mask=offs_am < M, other=1.0)[:, None] + tl.load(E + offs_am, mask=offs_am < M, other=0.0)[:, None]

    output_offs = offs_am[:, None] * output_row_stride + offs_bn[None, :]
    tl.store(C + output_offs, c.to(tl.float32), mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def fused_matmul_layer_norm(a, b, weight, bias, eps):
    M, K = a.shape
    K, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    _kernel_matmul_layer_norm[grid](
        a, b, out, weight, bias,
        M, N, K,
        eps,
        out.stride(0),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        weight.stride(0), bias.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out