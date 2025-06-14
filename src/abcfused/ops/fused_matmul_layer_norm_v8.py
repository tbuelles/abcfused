# type: ignore

import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(
    A, B, C, D,  # pointers to matrices
    M, N, K,  # matrix dimensions
    eps, # layer norm epsilon
    stride_am, stride_ak,  # strides layout
    stride_bm, stride_bk,
    stride_cm, stride_ck,
    stride_dm, stride_dk,
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
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bm

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = c_offs_m < M
    mask_n = c_offs_n < N
    c_ptrs = C + c_offs_m[:, None] * stride_cm + c_offs_n[None, :] * stride_ck
    tl.store(c_ptrs, accumulator, mask = mask_m[:, None] & mask_n[None, :])

    #Layer Norm part
    mean = tl.sum(accumulator, axis=1) / N
    variance = tl.sum((accumulator - mean[:, None])**2, axis=1) / N
    norm = (accumulator - mean[:, None]) / tl.sqrt(variance[:, None] + eps)
    d_offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    d_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = d_offs_m < M
    mask_n = d_offs_n < N
    d_ptrs = D + d_offs_m[:, None] * stride_dm + d_offs_n[None, :] * stride_dk
    tl.store(d_ptrs, norm, mask = mask_m[:, None] & mask_n[None, :])

def fused_matmul_layer_norm(a, b, eps):
    M, K = a.shape
    K, N = b.shape
    C = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    D = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    _kernel[grid](
        a, b, C, D,
        M, N, K,
        eps,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return D